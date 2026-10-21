# Benchmark status

No completed Diffusers generation benchmark is available yet. Initial strict TP6 loading is approximately 18 seconds with 7,163 MB of parameter storage per rank after FP32-sensitive text projection. Initial failed numeric probes are retained in data/probes and must not count as successful video throughput.

`scripts/benchmark.sh` submits real jobs to the independent API and writes JSON, CSV and Markdown, recording wall latency, success rate, videos per hour and generated seconds per hour. Completion is not a quality rating: usable-video proportion requires visual/audio review.

Required progression: one finite forward; 512×320/124f cold and cache-warm T2VA; one image reference; TP3; dual TP3; then higher resolutions and 10/20-job workloads only for modes passing memory and correctness gates.

For comparison, recent LightX2V 864×480/124f/20-forward T2VA jobs took approximately 303–307 seconds (TE ~40 s, DiT ~214–216 s, decode ~46–53 s). This is a different resolution from the first Diffusers probe. Match input, geometry, forward count and cold/warm conditions before claiming a speedup.
