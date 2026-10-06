# `20-profile.sbatch` — Nsight 剖面

**跑過：** 幾次，在需要知道「時間花在哪個 kernel」時。

比 per-stage histogram 細一層。**但多數時候 histogram 就夠了** ——
「TTFT 的 32% 是 KV 傳輸」是 `--enable-metrics` 免費給的，不需要 Nsight。
