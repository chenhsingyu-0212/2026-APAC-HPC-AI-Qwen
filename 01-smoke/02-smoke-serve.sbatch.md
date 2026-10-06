# `02-smoke-serve.sbatch` — 引擎能不能服務這份 checkpoint

**跑過：** 多次，每次換引擎或換叢集都跑。

載模型、起 server、`curl /health`、送一個請求。**不跑 AIPerf。**

**它抓到的：** `find_cuda_home()` 的斷言（job 79370，15 秒就死）—— sglang 的 deep_gemm
依序找 `CUDA_HOME`、`CUDA_PATH`、`which nvcc`、`/usr/local/cuda`，四個都沒有就斷言失敗。
NYCU 沒有 `/usr/local/cuda`。修法是指向 wheel 自帶的 `site-packages/nvidia/cu13`。
