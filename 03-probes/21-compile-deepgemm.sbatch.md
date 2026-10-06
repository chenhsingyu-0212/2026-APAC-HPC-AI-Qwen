# `21-compile-deepgemm.sbatch` — 預先填 DeepGEMM 的 JIT cache

**跑過：** job 86188（被 health check 在 155 秒殺掉）、215218（成功）。

**它存在的原因：** 開 `--moe-runner-backend deep_gemm` 之後引擎說
`Entering DeepGEMM JIT Pre-Compile session. It may take a long time (typically 10-20 mins)`，
而 health check 的期限比那短。DeepGEMM 從來沒有壞，它在編譯。

**它蓋不到的（註解裡寫清楚了）：** 這支工具編的是 **dense** FP8 路徑 ——
job 215218 產出 28 個 `sm90_fp8_gemm_1d2d`，lock 名字是 `_g1`（group of one）。
`--moe-runner-backend deep_gemm` 叫的 **MoE grouped GEMM 依真實 batch 的 expert 與
token 形狀 keyed，不在這裡編**，所以服務時仍然會 JIT。

**cache 在 `$HOME/.cache/sglang/deep_gemm`**，不是 `$HOME/.cache/deep_gemm` ——
後者是這支腳本在 215218 之前報錯的位置，讓一次成功的編譯看起來像失敗。
`$HOME` 在兩個叢集都是共用儲存，所以跑一次所有後續 job 都受益 —— **也就是說
我們所有成績都是暖 cache 的**，而主辦單位的新帳號不是。
