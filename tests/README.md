# `tests/` — 24 支測試，以及每一支是為了哪個已經發生過的錯誤

這些從來沒有送成 job。它們全部在本機跑，**而每一支都對應一個真的失敗過的東西** ——
所以它們是「我們犯過什麼錯」的清單，倒過來讀。

```
cd /mnt/nfs/work/hsingyu/HPC-AI && for t in run/test-*.sh; do bash "$t"; done
```

目前 **24 支全綠**。`test-topk-ids.sh` 本身要 96 秒，所以跑整套要留 5 分鐘以上。

| 測試 | 它擋住的那個錯誤 |
|---|---|
| `test-sched.sh`（21 項） | NYCU 的第二個 step 起不來（job 86436：prefill 健康，decode 卡在 `Requested nodes are busy`） |
| `test-port-guard.sh`（16 項） | **port 9100 是 Prometheus IPMI exporter**，吃掉 372421 和 372456 兩個實驗 |
| `test-site-contract.sh`（26 項） | `AIPERF` 不是 `AIPERF_BIN`（372918）。而它第一版誤報 firmus —— 裸名 `aiperf` 配容器是合法的 |
| `test-singularity-prefix.sh`（14 項） | `declare` 在被函式 source 的檔案裡變區域變數（372975）。**每一項都走 `sched_load`**，因為頂層 source 會過而 production 不會 |
| `test-gpu-arg.sh`（15 項） | `printf "--gpus-per-node=%d"` 被 bash 當選項；而第一版預設值**弄壞了 NYCU** 的 `gpu:gpu:N` |
| `test-prewarm.sh`（21 項） | 暖機只暖 n 條路由而不是 n 對 prefill–decode |
| `test-extra-flags.sh`（21 項） | per-role 的 extra flags 沒傳進去 |
| `test-mem-fraction.sh`（11 項） | per-role 的 mem fraction 沒傳進去（這是 +6.0% 的載體） |
| `test-install-moe-config.sh`（23 項） | MoE config 的三個地雷：`triton_version` key、`SGLANG_MOE_CONFIG_DIR` 取代整棵樹、檔名裡的空格 |
| `test-down-config.sh`（11 項） | `_down.json` 必須和 up 數值相同、只多 `USE_TMA` |
| `test-probe-mem.sh`（20 項） | 記憶體探針的解析 |
| `test-topk-ids.sh`（14 項） | sep tuner 的 topk id 產生 |
| `test-tune-sep.sh`（28 項） | sep tuner 的前置（而真正的失敗是它跑完 exit 0 卻零檔案，測試擋不住那個） |
| `test-env-quoting.sh`（10 項） | launcher 寫未引號的 `export`，讀它的 shell 把 JSON 大括號展開（216211 的 nixl 格） |
| `test-reap-quoting.sh` | reap 腳本的引號 |
| `test-nixl-env.sh`（7 項） | NIXL 參數的 JSON 組裝不能含空白 |
| `test-pbs-gpuids.sh` | UUID → **位置**（不是 `nvidia-smi` 的 index）的轉換 |
| `test-ep-optin.sh` | expert parallelism 必須是 opt-in —— 它預設開著，關掉值 2.27× |
| `test-prefill-graph.sh` | prefill 必須關 CUDA graph（DeepGEMM 的 module load） |
| `test-sglang-ref.sh`（8 項） | 我們的 argv 對得上參考部署 |
| `test-sglang-knobs.sh` | 旗標名在這個版本還存在 |
| `test-vllm-flags.sh` / `test-vllm-proxy.py` | vLLM 的旗標與那支自己寫的 gateway |
| `test-prepare-order.sh` | 前置檢查必須在任何 GPU 工作之前（215695 在載完模型後才發現 aiperf 路徑錯） |
| `test-teardown.sh` | 拆除要真的放開 GPU |
