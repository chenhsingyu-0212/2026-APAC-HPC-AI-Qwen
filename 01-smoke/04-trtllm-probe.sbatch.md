# `04-trtllm-probe.sbatch` — TensorRT-LLM 的 CLI 長什麼樣

**跑過：** job 17604（2026-09-16）。產出存在 `results/trtllm-help-17604/`。

把 `trtllm-serve --help`、`trtllm-serve disaggregated --help`、`llm_args.py` 的欄位
抓下來，**因為不能憑記憶寫 TensorRT-LLM 的設定**。

**它確立的：** `cache_transceiver_config` 是 **LLM args 的欄位**而不是 disagg config 的
（`llm_args.py:1807`）；`--server_role` 的值；傳輸後端預設是 **UCX** 而不是 NIXL。
換版本之後要重跑。
