# `lib/` — 每個 job 都 source 的那些

這些不是「跑過的腳本」，是**每一支跑過的腳本都依賴的東西**。放進來因為少了它們
上面任何一支都動不了。

| 檔案 | 它負責什麼 | 值得知道的 |
|---|---|---|
| `pd-lib.sh` | PD 的佈局、啟動、拆除、GPU 池 | `pd_assert_ports_free()` 在載入 checkpoint 前實測綁定 —— port 9100 吃掉過兩個實驗。`pd_assert_prefill_batch()` 在 DeepGEMM 把批次砍小時讓那一格失敗 |
| `sched-lib.sh` | 排程器的 seam | `sched_load()` 是**函式**，它 source backend —— 所以 backend 裡用 `declare` 宣告的陣列會變成它的區域變數。job 372975 就死在這 |
| `sched/slurm.sh` | Slurm 後端 | `SLURM_STEP_VIA=overlap` 避開 step 層級的 `--gres`（Nano5 必須）。`sched_sing_args()` 是**呼叫時**求值的 singularity 前綴 |
| `sched/pbs.sh` | PBS 後端 | PBS 沒有 per-step GPU 分配器，所以那段隔離在這裡是程式碼。UUID → 整數的轉換在這裡 |
| `site-lib.sh` | 叢集差異的載入 | `site_gpu_arg()` 回傳**整個** GPU 參數而不只是 `--gres` 後面的值 —— Nano5 要 `--gpus-per-node`。`printf --` 是必要的，否則 bash 把 `--gpus...` 當自己的選項 |
| `sites/nscc.sh` | NSCC（PBS，128 SU/GPU-hour，**停用中**） | UUID `CUDA_VISIBLE_DEVICES` 值 6.3% |
| `sites/nano5.sh` | Nano5（原生 Slurm，192 SU/16 卡時，可儲值） | 三條強制規則 ＋ port 9100 ＋ `/tmp` vs `/work` 的 118 倍，全部在註解裡附實測數字 |
| `sites/nycu.sh` | NYCU（**16 卡在政策上不可能**） | `defq-qos` 的 GrpTRES 是 `gres/gpu=15`，全組共用 |
| `sites/firmus.sh` | Firmus（H200，17 分鐘硬上限） | 141 GB 讓 prefill TP2 成為可能 |
| `engines/sglang.sh` | SGLang 的 argv | per-role 的 extra flags、mem fraction、NIXL 參數組裝。`KV_CACHE_DTYPE` 兩個 role 都接了 |
| `engines/vllm.sh` | vLLM 的 argv | `--kv-transfer-config` 的 JSON，角色叫 `kv_producer`/`kv_consumer` |
| `engines/vllm-pd-proxy.py` | **vLLM 沒附的那個 gateway** | 握手是兩步 POST，decode 去拉。**必須逐 chunk 轉發** —— 緩衝的 proxy 會讓 TTFT 吸收整段生成、每個 ITL 塌到 0，計分指標變成在量這支檔案 |
| `engines/trtllm.sh` | TensorRT-LLM 的 argv | 五個 pip 不會安排的前置，全部在註解裡。`moe_config.backend` 有靜默退回 |
| `engines/lmdeploy.sh` | LMDeploy | **沒有 PD 模式**，所以 `eng_pd_worker_cmd` 刻意不存在 |
| `client-lib.sh` | AIPerf 的 argv | `client_init` 用 `aiperf profile --help` 偵測這個版本接受什麼旗標。暖機的 burst/sequential 兩階段在這裡 |
| `engine-lib.sh` | 引擎無關的那層 | `eng_image()` 回傳 `$SGLANG_IMG` —— 所以 `.sif` **不能**放那裡，它會被餵給 pyxis |
| `reap-gpus.sh` | 收掉上一格還握著 GPU 的進程 | 沒有它，下一格會在載入 160 秒後死於 `The memory capacity is unbalanced` |
| `results-row.sh`, `campaign-summary.py`, `profile-summary.py` | 把產出變成可讀的表 | campaign summary 的 RANKED / BELOW THE GATE 兩段就是這裡來的 |
| `check-basic.sh` | 檢查這次跑的符不符合 Basic | 純 PD、FP8 以上、無損 |
| `venv-prepare.sh` | 沒有容器時的 pip venv | NYCU 唯一的路 |
