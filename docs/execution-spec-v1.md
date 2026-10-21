# IBM AC922 · Diffusers H3 高吞吐推理服务实施方案

Agent 执行规格 v1.0

核心目标：在 IBM POWER9 + 6×V100 16GB 环境下，使用 Diffusers 构建速度优先、并发优先的 MiniMax H3 视频生成服务。

本方案可以直接交给 Codex、Claude Code 或其他编程 Agent 执行。

## 一、总体技术决策

我建议采用：

Diffusers Modular Pipeline + 组件级显存管理 + 双三卡 Worker + 任务队列 + 可切换 TP6。

但这里有一个关键区别：

> 双 TP3 是最终希望实现的并发架构，不是已经证明可行的配置。Agent 必须先证明单组三卡能容纳 H3 并完成推理，再将双 Worker 设为默认。

你的两组 GPU 正好对应两个 NVLink 域：

AIserver · FastAPI

统一 API / 任务队列 / 资源调度 / 结果管理

GPU Scheduler

独占 GPU 组 · 显存预算 · 优先级队列

Worker A

CUDA_VISIBLE_DEVICES=0,1,2

Diffusers H3

TP3 / Offload

GPU 0

GPU 1

GPU 2

Worker B

CUDA_VISIBLE_DEVICES=3,4,5

Diffusers H3

TP3 / Offload

GPU 3

GPU 4

GPU 5

VAE / FFmpeg / 结果存储

按显存预算执行，避免解码挤占另一个 Worker

目标架构：两个独立三卡 Worker。若 TP3 不可行，则使用 TP6 单 Worker，并保留独立预处理与编码队列。

### 为什么不直接使用六卡 TP6？

你目前 TP6 已经稳定运行，但有两个性能限制：

1. GPU0–2 与 GPU3–5 跨组通信走 PCIe/SYS。
2. 六张 GPU 被同一个任务独占，无法同时生成第二条视频。

双 TP3 的目标是让两个任务并行运行，并减少跨组的 collective communication。

但必须考虑单组三卡只有 48GB 显存，而且并非统一显存。H3 的权重切分、激活、TE 和 VAE 必须全部适配。

## 二、基础技术选型

| 层级           | 方案                                      | 原则           |
| ------------ | --------------------------------------- | ------------ |
| 推理框架         | Hugging Face Diffusers Modular Pipeline | 固定已验证 commit |
| 模型           | MiniMax H3                              | 复用本地已有权重     |
| 主精度          | FP16                                    | V100 原生支持    |
| 权重压缩         | INT8 优先测试                               | 解决 TP3 显存限制  |
| Attention    | PyTorch SDPA                            | 先建立兼容基线      |
| 高速 Attention | 现有 H3 V100 FlashAttention               | 后续接入与验证      |
| 并行           | TP3 × 2 / TP6 × 1                       | 自动选择经验证的模式   |
| Offload      | Diffusers block-level group offload     | 仅用于无法驻留的组件   |
| 加速           | H3 Turbo 4-step / 8-step                | 依据模式选择       |
| API          | 现有 AIserver / FastAPI                   | 不重复开发网关      |
| 任务状态         | SQLite                                  | 持久化、恢复       |
| 进程管理         | systemd --user                          | 与现有部署保持一致    |

参考实现：

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers?utm_source=chatgpt.com)
- [Diffusers_minimax-h3](https://github.com/animede/Diffusers_minimax-h3?utm_source=chatgpt.com)
- [OpenVDN H3 Diffusers 实现](https://github.com/OpenVDN/vdn-minimax-h3?utm_source=chatgpt.com)

社区低显存 Diffusers 实现已经在单张 16GB RTX 4060 Ti 上完成 H3 T2VA，但是完整任务约需 25 分钟；其低显存方案主要依靠量化、投影 TE 和 block-level offload。这个结果说明有适配空间，但不能直接套用它的速度数据或认为 TP3 一定可行。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 三、性能优化策略

### 1. 第一目标：减少实际 DiT 计算次数

首先接入官方 Turbo 4-step 和 8-step。

建议分为：

| 模式        | 步数            | 用途   | 默认      |
| --------- | ------------- | ---- | ------- |
| `turbo4`  | 4 次实际 forward | 批量预览 | 优先速度    |
| `turbo8`  | 8 次实际 forward | 动画制作 | 质量/速度平衡 |
| `quality` | 已验证的高质量配置     | 重要镜头 | 按需      |

注意 Diffusers 不同提交中的 `num_inference_steps` 语义可能涉及终止 sigma 的计数差异。Agent 必须记录实际模型 forward 次数，不能直接复制 LightX2V 的 `steps+1`。

某些 H3 Diffusers 实现曾把 8 次模型评估对应到 9 个 sigma grid 点。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



优化的衡量单位是实际 DiT forward 次数，而不是 API 中传入的 steps 数字。

### 2. 第二目标：优先让 DiT 常驻显存

Diffusers 支持 block-level group offloading，可以将部分 transformer blocks 保留在 CPU，然后按需加载到 GPU。它可以节省显存，但会增加数据传输。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



因此建议采用三个执行等级：

| 执行等级     | 显存方式                      | 适用情况  |
| -------- | ------------------------- | ----- |
| Resident | DiT 分片常驻 GPU              | 最快，优先 |
| Hybrid   | 部分层卸载 CPU                 | 显存紧张  |
| Streamed | block-level group offload | 低显存保底 |

优先选择 Resident，而不是默认打开 group offload。

POWER9 CPU 和 GPU 间频繁传输大量权重可能成为严重瓶颈。

Agent 应记录每个 denoise step 的 CPU→GPU 传输时间、GPU 空闲时间和实际运算时间。

另外，Diffusers 曾出现影响 H3 group offloading 的版本回归，因此必须锁定成功验证的 commit，不能直接依赖最新版 `main`。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### 3. 第三目标：优化文本编码阶段

当前基线中：

- T2VA conditioning 约 38–42 秒。
- Ref2AV conditioning 约 95 秒。

建议将 TE 作为独立执行阶段。

实现：

- 相同 prompt、参考图和模型配置使用条件缓存。
- TE 和 DiT 分阶段管理显存。
- 减少不同任务之间的模型反复加载。
- 对不同编码器版本生成独立缓存命名空间。
- 测试现有 INT8 TE，但不要未经质量验证直接替换原版。

缓存键必须包含 tokenizer、TE 权重版本、参考输入、提示词以及影响 conditioning 的全部参数。

当两个 Worker 并发时，TE 不得无控制地申请另一个 Worker 的 GPU。

### 4. 第四目标：GPU 分组与 NUMA 亲和

建议 Agent 实测当前 GPU 拓扑：

```
nvidia-smi topo -m
numactl --hardware
```

优先让 Worker A、B 各自绑定对应的 CPU 内存域和 GPU 组。

注意你的报告同时出现“8 个 NUMA 节点”和 CPU 亲和表述，因此必须以系统实际 NUMA 编号为准，不得直接硬编码 node0/node8。

测试：

- TP6 跨组推理。
- 单组三卡 TP3。
- 双 TP3 并发。
- 一个 TP3 执行时另一个 GPU 组执行 TE/VAE。
- CPU offload 开启和关闭的差异。

真正的目标不是 GPU 使用率 100%，而是每小时完成更多有效视频。

### 5. 第五目标：VAE 不阻塞下一条任务

你的 H3 当前 VAE 解码耗时约 30–60 秒。

可以研究：

- DiT 生成完成后释放必要显存。
- VAE 在可用 GPU 上执行。
- FFmpeg 使用独立 CPU 进程。
- 在 GPU 资源允许时，下一条任务与上一条任务的编码阶段重叠。

但不要直接假定某张 V100 可以一直留给 VAE，否则可能破坏 TP3 推理的显存预算。

VAE 应由统一 GPU Scheduler 安排。

# 四、可以直接交给 Agent 的任务说明

下面是我建议的完整执行指令。

# 任务：IBM AC922 Diffusers MiniMax H3 高吞吐推理服务

## 1. 项目目标

将 IBM AC922 上的 MiniMax H3 视频推理引擎迁移到 Hugging Face Diffusers Modular Pipeline。

**主要优化目标：速度优先、并发优先，同时保证生成结果正确、服务长期稳定。**

最终希望在 6×V100 16GB 上实现两个并发 H3 任务。如果不可行，需要自动回退到经过验证的 TP6 单任务模式，并说明实际原因。

禁止通过牺牲基本视频正确性、破坏音视频同步或不受控降低画质来伪造性能提升。

## 2. 当前机器

- IBM Power Systems AC922
- CPU：2×POWER9，ppc64le
- 内存：126 GiB
- GPU：6×NVIDIA Tesla V100 SXM2 16GB
- GPU 0–2 为一组 NVLink
- GPU 3–5 为一组 NVLink
- 跨组走 PCIe/SYS
- Debian 12
- NVIDIA Driver 535.216.03
- PyTorch 2.10.0，CUDA 12.4
- 已有 LightX2V H3 TP6 推理链路
- 已有 V100 H3 FlashAttention 编译模块
- 已有 AIserver / FastAPI
- 已有 H3 BF16、FP8、INT8 TE、VAE 和 Turbo LoRA 权重
- 模型目录位于 /srv

现有 LightX2V 是已验证的基线和回退服务，不允许破坏或覆盖。

## 3. 实施原则

1. 必须使用 Diffusers 作为新推理引擎。
2. 不允许直接修改现有生产 Python 环境。
3. 创建独立 venv 和运行目录。
4. 复用已有模型权重，不要盲目下载重复的 H3 全量模型。
5. 固定 Diffusers、Transformers、Accelerate 等依赖的精确版本。
6. 禁止直接使用针对 sm_80、sm_89、sm_90、sm_120 的专用二进制。
7. 优先使用 PyTorch SDPA 作为基线；V100 FlashAttention 必须单独验证后启用。
8. 不允许默认使用 BF16 CUDA 计算；根据 V100 能力设置 FP16 与 FP32 敏感算子。
9. 不得把六张 16GB GPU 当成一张 96GB GPU 使用。
10. 对未经验证的量化、LoRA 和 offload 配置，必须标记为实验模式。

## 4. 阶段一：环境审计

检查以下内容：

- POWER9 与现有 PyTorch CUDA 环境
- Diffusers H3 实现所需依赖
- 各二进制扩展的架构兼容性
- GPU P2P / NCCL 拓扑与可用性
- H3 本地权重结构与 Diffusers 组件格式
- 已有模型是否需要转换或重新分片
- 可用磁盘空间与内存容量
- 现有 AIserver API 和任务状态逻辑

输出 `docs/environment-audit.md`。

严禁在未检查存储和权重格式的情况下复制 199GB 模型目录。

## 5. 阶段二：最小 Diffusers H3 推理链路

建立：

- T2VA 文生音视频
- 图像参考 I2VA / Ref2VA
- 音频和视频 VAE 解码
- MP4 输出
- 单次任务完整性能记录

首先使用 PyTorch SDPA 和官方采样设置。

验证相同输入能生成符合预期的视频、音频和时长。

若原生 Diffusers 无法在 V100 上直接加载或推理，应记录明确错误，并在独立适配层解决。

不得静默回退到 LightX2V 冒充 Diffusers 成功。

## 6. 阶段三：GPU 并行方案

实施并测试以下候选模式：

A. Diffusers TP6 单任务

B. Diffusers TP3 单组三卡

C. Diffusers 双 TP3 双任务

D. Diffusers + 部分 CPU offload

E. Diffusers + block-level group offload

必须注意：Diffusers 不能仅靠配置 `CUDA_VISIBLE_DEVICES=0,1,2` 就自动实现 TP3。需要验证真实的分布式切分实现、通信与模型分片。

不能简单将每张 GPU 加载完整 DiT 误认为 tensor parallel。

对每种模式记录：

- 是否成功运行
- 每卡峰值显存
- CPU 内存峰值
- 权重加载时间
- Conditioning 时间
- DiT 总时间
- 每次 DiT forward 平均时间
- VAE 解码时间
- 端到端时间
- 是否出现 OOM、NaN 或音视频异常

若 TP3 不可行，不允许为了宣称双并发而无限增加 CPU offload，必须比较实际吞吐量后决定是否使用 TP6。

## 7. 阶段四：模型加速

按顺序执行：

1. 官方 Turbo 4-step 配置
2. 官方 Turbo 8-step 配置
3. SDPA 与 V100 H3 FlashAttention A/B
4. FP16 数值安全检查
5. INT8 权重量化
6. DiT 常驻显存策略
7. block-level offload
8. Conditioning 缓存
9. VAE 分阶段执行与 FFmpeg 异步编码

所有优化必须独立开关，不能一次性修改全部参数。

每个配置至少运行多个固定 seed，并确认输出可用。

## 8. 阶段五：高并发任务调度

在现有 AIserver 上增加统一推理调度：

- SQLite 持久化任务队列
- GPU 组独占锁
- Worker 健康检查
- 模型加载状态
- GPU 显存水位线
- 最大运行任务数
- 最大排队数
- 超时处理
- Worker 异常恢复
- 用户取消任务
- 任务失败原因记录
- 结果文件自动清理

同时防止两个 Worker 竞争同一 GPU 组。

生产环境不得依靠简单的进程数量配置来提高并发。

## 9. 阶段六：性能基准

至少测试以下任务：

- 512×320，124 帧
- 1024×576，124 帧
- 1280×768，124 帧
- T2VA
- Ref2VA / I2VA

测试任务数量：

- 单任务冷启动
- 单任务热态
- 连续 10 个任务
- 两任务同时提交
- 连续 20 个混合任务

其中 1280×768 等规格必须先通过模型与显存可行性检查，不得盲目并发。

对每个测试输出 JSON、CSV 和 Markdown 性能报告。

核心指标：

- 每小时成功视频数量
- 每小时成功生成的视频总秒数
- 平均任务耗时
- P50/P95 任务耗时
- GPU 峰值显存
- GPU 利用率
- CPU 内存占用
- CPU↔GPU 数据传输耗时
- 任务成功率
- 可用视频比例

需要比较新 Diffusers 与原 LightX2V。

## 10. 上线验收

必须满足：

1. Diffusers H3 确实完成端到端生成。
2. T2VA 基础链路成功。
3. 参考图生成链路成功。
4. 视频及音频解码正确。
5. 系统重启后任务状态可恢复或明确失败。
6. 不会因为两个任务同时提交而抢占同一组 GPU。
7. OOM 或 Worker 崩溃不会导致整个 API 不可用。
8. 可以一键切回原 LightX2V。
9. 存储空间不足时拒绝新任务。
10. 完整输出性能报告和已知限制。

目标性能：

- 争取双 TP3 并发，但不是强制验收条件。
- 单任务速度尽量不低于现有 LightX2V。
- 优先优化总吞吐量，而不是仅优化单任务耗时。
- 连续 20 个测试任务不得发生服务级不可恢复故障。
- 基本生成质量不得明显低于已有基线。

## 11. 交付文件

至少输出：

- `docs/environment-audit.md`
- `docs/architecture.md`
- `docs/compatibility.md`
- `docs/benchmark.md`
- `docs/optimization-results.md`
- `docs/known-limitations.md`
- `config/quality.yaml`
- `config/turbo4.yaml`
- `config/turbo8.yaml`
- `scripts/benchmark.sh`
- `scripts/start-worker-a.sh`
- `scripts/start-worker-b.sh`
- systemd Worker 服务配置
- 任务队列和 Worker 管理代码
- 自动化测试和性能记录脚本

需要给出明确的生产推荐配置与回退配置。

不得把未经验证的候选方案描述为已经实现。

## 12. 决策要求

最终必须用实测数据回答：

1. Diffusers TP6 是否比原 LightX2V 更快？
2. 单组三卡 TP3 是否可行？
3. 双 TP3 是否真正提升每小时产量？
4. INT8 量化带来的速度与质量影响有多大？
5. Offload 是否使模型传输成为主要瓶颈？
6. Turbo 4-step 和 8-step 哪种更适合作为默认服务？
7. 当前硬件的最佳生产部署配置是什么？

允许 Agent 根据实际测量结果调整技术路线，但所有重大变更必须记录原因、结果与回退方法。

## 五、Agent 应该如何决定最终 GPU 架构

我不建议一开始就强制双 TP3。

应该用测试结果自动选择。

架构决策模拟

单组三卡 TP3 能否成功完成 H3？

尚未测试

可以，且显存稳定

不可以，发生 OOM 或执行失败

建议部署模式

保留现有 TP6，优先完成验证

双 TP3 未经证实，不应直接部署到生产。

## 六、性能目标如何设置

按照你已有的 LightX2V 基线，5.17 秒、1280×768 的 8 步视频约需要 314 秒。

不同吞吐目标对应的产能

以下仅是假设每条任务都成功、可持续稳定运行的计算示例，不是 Diffusers 实测值。

每小时完成的 5.17 秒视频数量，按指定时长和并发数估算。

020406080现有 314s ×1240s ×1240s ×2180s ×2120s ×2

这张图说明为什么我支持你追求并发，但必须区分两个概念：

- 延迟：用户提交一条视频，多久能拿到。
- 吞吐量：服务器持续运行，每小时能完成多少条合格视频。

对于一个多人共享的视频生成服务，如果双 TP3 让单任务从 200 秒变成 280 秒，但整体吞吐翻倍，依然可能是更好的业务选择。

不过如果 TP3 大量使用 CPU offload，单任务耗时变成 800 秒，那么双并发可能还不如 TP6。

## 七、我认为最重要的两个技术风险

### 风险 1：Diffusers 并不自动具备高效 TP3

这是整个项目最大的技术不确定性。

Diffusers 提供 pipeline、offload 和模型组件管理能力，但不代表当前 H3 的各模块已经具备针对六张 V100 的现成 TP 实现。

Agent 需要检查 transformer 的 attention head、hidden dimension、FFN、联合音视频路径、KV 组织和相关权重布局是否适合 TP3。

不能简单复用 LightX2V 已有的 TP6 shard，也不能仅使用 `device_map` 认为获得了真正的张量并行。

如果需要大幅修改 H3 transformer 的 forward 逻辑，项目成本会明显增加。

### 风险 2：126GiB CPU 内存可能不足以支撑两个完整模型副本

H3 原始 BF16 权重和大 TE 占用很高。

同时启动两个 Worker，如果各自保留完整 CPU 权重副本，可能导致主机内存紧张、swap 频繁使用，甚至 OOM。

建议：

- 尽可能使用共享只读权重文件和内存映射。
- 分析进程的 PSS，而不是简单累加 RSS。
- CPU offload 与量化结合测试。
- 不允许两个 Worker 无限制缓存完整 TE。
- 始终保留系统、任务队列和服务所需的内存余量。

特别是社区 Diffusers H3 低显存方案表明，block-level offload 虽然能降低显存占用，但可能让权重传输成为主要成本。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 八、最终执行建议

建议 Agent 将项目拆成三个明确里程碑：

里程碑 1：Diffusers 单任务跑通

不追求并发。目标是完整兼容 POWER9、V100、FP16、H3，并获得可信的单任务性能数据。

里程碑 2：验证 TP3 和双 Worker

研究是否能独立使用 0–2 和 3–5 两组 GPU，比较 TP6、TP3 与 CPU offload 的实际吞吐。

里程碑 3：速度优化和生产部署

加入 Turbo、缓存、内核优化、队列与自动恢复，选择每小时合格视频产量最高的配置。

我的首选生产目标是：双组三卡，各运行一个常驻的 Diffusers H3 Turbo Worker。

但是，有一个重要的性能原则：不要为了双并发而接受极慢的逐层 CPU offload。 如果单组三卡装不下高效运行所需的模型分片，就让六张卡继续共同运行一个推理任务，把优化重点转向 Turbo 4-step、模型常驻、conditioning 缓存，以及减少 VAE/编码等待。

最终通过基准测试决定架构，而不是预先决定必须双 Worker。