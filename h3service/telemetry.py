"""Bounded host/GPU measurements for real inference runs."""
import subprocess
import threading
import time

import psutil


class Monitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak_pss = 0
        self.peak_gpu = {}
        self.util_sum = {}
        self.util_count = {}
        self.samples = []
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        parent = psutil.Process()
        while not self.stop_event.is_set():
            try:
                processes = [parent, *parent.children(recursive=True)]
                pss = 0
                for p in processes:
                    try:
                        pss += p.memory_full_info().pss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_pss = max(pss, self.peak_pss)
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                               "--format=csv,noheader,nounits"], text=True, timeout=5)
                rows = []
                for line in raw.splitlines():
                    i, memory, util = [int(s.strip()) for s in line.split(",")]
                    self.peak_gpu[i] = max(memory, self.peak_gpu.get(i, 0))
                    self.util_sum[i] = self.util_sum.get(i, 0) + util
                    self.util_count[i] = self.util_count.get(i, 0) + 1
                    rows.append([i, memory, util])
                if len(self.samples) < 4096:
                    self.samples.append({"seconds": time.perf_counter() - self.started, "gpus": rows, "process_tree_pss_bytes": pss})
            except (OSError, ValueError, subprocess.SubprocessError, psutil.Error):
                pass
            self.stop_event.wait(2)

    def start(self):
        self.thread.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.finish()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        return {"peak_process_tree_pss_bytes": self.peak_pss, "peak_gpu_used_mib": self.peak_gpu,
                "mean_gpu_utilization": {i: self.util_sum[i] / self.util_count[i] for i in self.util_sum},
                "samples": self.samples}
