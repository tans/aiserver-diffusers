# AIserver Diffusers

Independent MiniMax H3 inference service for IBM POWER9 and six V100 16 GB GPUs. Work in progress: no inference configuration is production verified yet.

The existing LightX2V API and Python environment are preserved. The public API will retain the existing `/v1/models`, `/v1/files`, `/v1/jobs`, status, cancellation and result contracts. Image generation and super resolution are initially out of scope.

See [the execution specification](docs/execution-spec-v1.md) and [environment audit](docs/environment-audit.md). All source changes are committed on `main` and pushed. Models and task media are not stored in Git.
