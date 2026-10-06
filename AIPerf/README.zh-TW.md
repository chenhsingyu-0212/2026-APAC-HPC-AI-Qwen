# AIPerf 量測工具組

> English version: [README.md](README.md)

四支腳本，讓 LLM 推論服務的數字在**不同機器、不同叢集、不同推論引擎之間可以直接並排比較**
（vLLM / SGLang / Dynamo / TRT-LLM —— 只要有 OpenAI 相容的 `/v1/chat/completions` 都可以）。

| 腳本 | 用途 |
| --- | --- |
| `install.sh` | 建 venv、裝 `aiperf`，並驗證這個版本實際支援哪些旗標。 |
| `run.sh` | 暖機、跑對齊點、掃描並行數，可選擇擾動 ISL/OSL，最後組出 `metrics.csv`。 |
| `gpu_sample.sh` | 掃描期間在 **server 節點**上跑：GPU 功耗與時脈、記憶體、PCIe/NVLink/InfiniBand、CPU，以及機器上還有誰。 |
| `diagnose.sh` | 讀完一輪的結果，直接告訴你**瓶頸在哪裡** —— 含 roofline，判定算力 vs 記憶體頻寬。 |
| `official.sh` | 2026 APAC HPC-AI 計分用的官方指令，一字不差，附 `--verify` 直接跟主辦方的規則檔對照。 |
| `observe.sh` | 監控一個不能被改動的指令 —— 從外面輪詢，產出 `diagnose.sh` 要吃的東西。 |

四支都不依賴這個 repo 裡的其他東西 —— 複製到任何地方都能單獨跑。

---

## 快速開始

```bash
chmod +x *.sh                   # 只有第一次需要

./install.sh                    # 建立 ./aiperf-venv
./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct
```

結果會放在 `./aiperf-<時間戳>/`，比較表在 `./aiperf-<時間戳>/metrics.csv`。

2026 APAC HPC-AI 的計分重跑則是指令鎖死、監控必須留在外面：

```bash
./official.sh --verify                       # 動手之前，先確認我們跟規則一致
./gpu_sample.sh --out ./tel                  # 每個 server 節點上
./observe.sh --out ./scored --gateway gw01:8000 -- \
    ./official.sh --host gw01:8000 --model qwen3-pd --concurrency 64
./diagnose.sh ./scored --telemetry ./tel --weights 22 --tp 8
```

---

## 1. `install.sh`

找到堪用的 Python → 建 venv → 裝 `aiperf` → **驗證並印出實際可用的旗標**。

最後一項才是重點：AIPerf 的序列長度旗標在不同版本有兩種寫法
（`--synthetic-input-tokens-mean/--output-tokens-mean` 對上 `--isl/--osl`），
`run.sh` 必須在執行時挑對的那一種。

```bash
./install.sh                            # 裝到 ./aiperf-venv
./install.sh --venv /work/$USER/venv    # 裝到指定位置
./install.sh --pin 0.12.0               # 釘住版本
./install.sh --check-only --venv PATH   # 只驗證現有安裝，不裝東西
./install.sh -h                         # 說明
```

| 旗標 | 預設 | 說明 |
| --- | --- | --- |
| `--venv PATH` | `$AIPERF_VENV`，否則 `./aiperf-venv` | 相對路徑是相對於**當前目錄**，不是腳本所在目錄。 |
| `--pin VERSION` | 不釘（裝最新） | 等同 `pip install aiperf==VERSION`。 |
| `--check-only` | 關 | 略過找 Python 與安裝，只跑驗證步驟。 |

**需求：** Python ≥ 3.11 且 < 3.14（AIPerf 自己宣告的範圍），以及 pip 能連外網。腳本會依序試
`python3.13 … python3.10`，最後才退回 `python3`；HPC 站台通常要先 `module load python`
（或用 conda），因為 `/usr/bin/python3` 常常還是 3.6。

**跨環境比較時請釘住版本。** 結尾的橫幅會直接印出該給隊友的 `--pin` 參數。

**注意 PATH 警告。** 如果 `PATH` 上已經有另一份 `aiperf`，腳本會提醒你。
一律用絕對路徑（`$VENV/bin/aiperf`），不要打裸的 `aiperf` —— `run.sh` 已經幫你這樣做了。

---

## 2. `run.sh`

```bash
./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct

./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct \
         --sweep "8 16 32" --isl 4000 --osl 200 --out ./myrun --label vllm-tp2
```

### 必填

| 旗標 | 說明 |
| --- | --- |
| `--url` | 服務的 base URL，例如 `http://node01:8000`。不要結尾斜線，也不要帶 `/v1`。 |
| `--tokenizer` | **本地**模型目錄（或檔案）。避免量測時上網抓，也確保 token 計數與服務端一致。 |

### 選填

| 旗標 | 預設 | 說明 |
| --- | --- | --- |
| `--model` | 向 `GET /v1/models` 詢問 | 只有在端點沒有模型清單時才需要（例如 TRT-LLM 的 disagg proxy —— 改問它其中一個 worker）。 |
| `--venv` | `$AIPERF_VENV`，否則 `./aiperf-venv` | 必須跟 `install.sh` 用的一致。 |
| `--isl` / `--osl` | `4000` / `200` | 合成的輸入／輸出序列長度（token 數）。 |
| `--sweep` | `"8 16 32 64 128 256"` | 以空白分隔的並行數，要用引號包成單一參數。 |
| `--isl-sweep` | 關 | 擾動維度：在固定並行數下要試的輸入長度。至少要 3 個值才構成斜率。 |
| `--osl-sweep` | 關 | 擾動維度：在固定並行數下要試的輸出長度。至少要 3 個值。 |
| `--shape-c` | 同 `--anchor` | 兩個 shape 掃描使用的並行數。 |
| `--anchor` | `32` | 全隊對齊點，會單獨再量一次。 |
| `--sla` | `"time_to_first_token:1000 request_latency:5000"` | `指標：門檻` 配對，門檻單位是**毫秒**。 |
| `--req-per-c` | `10` | 每個工作點的請求數 = `並行數 × 這個值`。想要百分位數更穩就調大。 |
| `--out` | `./aiperf-<時間戳>[-label]` | 輸出目錄。 |
| `--label` | 無 | 接在預設輸出目錄名後面，並記進 `env.txt`。 |
| `--no-warmup` | 開啟暖機 | 跳過暖機那一輪。 |
| `--metrics-url` | `<url>/metrics` | 要輪詢的 Prometheus 端點。探測失敗會自動關閉並警告。 |
| `--metrics-interval` | `2` | 輪詢週期（秒），必須 ≥ 0.1，避免輪詢本身干擾量測。 |
| `--no-metrics` | 開啟 | 完全不輪詢服務端。 |
| `-h`, `--help` | — | 印出檔頭說明區塊。 |

### 一輪跑下來的順序

0. **參數驗證** —— 開跑前先檢查所有旗標：URL 的 scheme、tokenizer 路徑是否存在、
   `--isl` / `--osl` / `--anchor` / `--req-per-c` 與每個 sweep 值是否為正整數、
   venv 裡有沒有 `aiperf`。參數打錯會在第一秒就失敗，而不是跑到二十分鐘後才發現。
1. **健康檢查** —— `GET /health`。沒回 200 只是警告不是失敗，有些服務沒有這個端點。
2. **模型 id** —— 沒給 `--model` 就向 `GET /v1/models` 問。絕不用猜的；
   模型名稱錯了整場會安靜地白跑。
3. **時鐘偏移** —— 用服務端自己的 HTTP `Date` 標頭跟本機時鐘比對，把差值記成
   `env.txt` 裡的 `server_clock_skew_s`。`points.csv` 的時間戳來自 client，
   `gpu_sample.sh` 的 telemetry 時間戳來自 server，兩台沒對時的話每個工作點都會切錯
   時間窗而且完全不會報錯。`diagnose.sh` 會依這個差值自動修正。
4. **旗標偵測** —— 挑出這個 AIPerf build 支援的序列長度寫法；
   缺 `--streaming` 就直接中止（沒有它就量不到 TTFT 與 ITL）。
5. **環境指紋** —— 寫出 `env.txt`（主機、URL、模型、tokenizer、各版本、所有參數、總請求數）。
   這份檔案就是事後兩輪能不能比較的關鍵，請跟結果一起留著。
6. **暖機** —— `c=8`、40 個請求，輸出寫到 `warmup.log`，且**不列入結果**
   —— 它在 `metrics.csv` 裡不會有任何一列。冷啟動前幾秒的延遲比穩態高 30–40%。
7. **對齊點** —— 用 `--anchor` 的並行數單獨跑一次。拿它跟 sweep 裡同樣並行數的結果比對，
   就知道量測本身穩不穩。
8. **掃描** —— 依序跑 `--sweep` 裡的每個並行數。
9. **Shape 掃描** —— 有給 `--isl-sweep` / `--osl-sweep` 的話，在 `--shape-c` 下跑。
   刻意排在最後：它們是額外證據，不該因為這裡失敗而賠掉其他東西都建立在上面的並行數曲線。
10. **組表** —— 解析所有 `profile_export_aiperf.csv` 併成一份 `metrics.csv`，
   並印出 goodput 最佳的工作點。

每一輪正式量測都強制使用 `--streaming` 與 `--endpoint-type chat`。

### 擾動掃描（`--isl-sweep`、`--osl-sweep`）

並行數掃描能做到的，永遠只是把瓶頸跟負載**關聯**起來。要判斷一個結論是不是真的，
必須去改動它所依賴的輸入，再確認反應符不符合那個結論宣稱的機制。這兩個旗標就是在做這件事
—— 一次只動一個 shape 維度，並行數固定，其他都不變：

```bash
./run.sh --url http://node01:8000 --tokenizer /models/Llama-3.1-8B-Instruct \
         --isl-sweep "1000 2000 4000 8000" --osl-sweep "50 200 800"
```

| 掃描 | 它讓什麼變大 | 該看的反應 |
| --- | --- | --- |
| `--isl-sweep` | prefill 工作量**以及**每步的 KV context | TTFT 應該跟 ISL 成正比；而 **ITL 的斜率**正好就是 roofline 的 KV 項，所以它能證實或推翻 `MEMORY BW` 這個判定。 |
| `--osl-sweep` | 只增加 decode 步數，其他都不變 | ITL 應該幾乎不動。如果它往上爬，代表每步成本並非固定 —— 通常是 KV cache 隨輸出變長而膨脹，那是容量壓力而不是穩態。 |

兩者預設都關閉；每個值會多花 `--shape-c × --req-per-c` 個請求。產出是 `isl<N>/` 與
`osl<N>/` 點目錄，`diagnose.sh` 會自動讀回來。每個點自己要求的 ISL/OSL 都記在
`points.csv` 裡，所以 shape 點是跟**它自己的目標值**比對，不是跟整輪的設定值比。

### 引擎內部指標（`engine_metrics.csv`）

如果服務端有 Prometheus 端點，`run.sh` 會在**每個工作點進行期間持續輪詢**，
並把取樣結果整理成 `engine_metrics.csv`，同時印出重點摘要：

```
  point           kv/token usage  queue peak  preemptions
  c8                        0.23           0            0
  c16                       0.93          90           88
```

| 訊號 | 結論 |
| --- | --- |
| KV／token 用量偏低且平坦 | 不是記憶體受限——可以調高 `gpu_memory_utilization` 或並行數。 |
| 排隊深度持續增長 | 請求在等而不是在跑：卡在排程器上限（`max_num_seqs`）。 |
| preemption 持續增加 | KV cache 超賣；隨之而來的延遲尖刺正是達標率崩掉的元兇。 |

**要在工作點「進行中」輪詢，不是前後各抓一次。** 排隊深度、cache 用量這類 gauge
在負載停止後就會歸零，前後抓完全看不出東西——所以這裡是持續取樣並回報峰值。

各引擎的端點：

| 引擎 | 端點 | 說明 |
| --- | --- | --- |
| vLLM | API 同 port 的 `/metrics` | 預設開啟。V0 與 V1 引擎的指標名稱不同。 |
| SGLang | `/metrics` | **啟動時要加 `--enable-metrics`**，否則根本沒東西可抓。 |
| TRT-LLM | Triton 走 port `8002`；`trtllm-serve` 走 `/metrics` | Triton 的情況要自己指定 `--metrics-url`。 |
| Dynamo | frontend 的 `/metrics` | `dynamo_*` 命名。 |

這個彙整是**刻意不綁定指標名稱**的——比對關鍵字時用的是完整的 `name{labels}`，
因為 Triton 把狀態放在 label 裡（`nv_trt_llm_request_metrics{request_type="waiting"}`），
而 vLLM 是放在名稱裡。原始的取樣內容會保留在每個工作點的 `metrics_samples.txt`，
就算過濾條件漏掉你在意的指標也不會遺失資料。

### 輸出結構

```
aiperf-20260817-143022-vllm-tp2/
├── env.txt          # 環境指紋 —— 請跟結果一起留著
├── warmup.log       # 暖機輸出（不列入 metrics.csv）
├── warmup/          # 暖機產出（組表時會跳過）
├── anchor-c32/      # 對齊點
│   └── metrics_samples.txt   # 這個工作點期間的原始 Prometheus 取樣
├── c8/ c16/ c32/ …  # 每個掃描點一個目錄
├── isl1000/ osl50/ …         # shape 掃描點（有跑的話）
├── points.csv       # 各工作點的並行數、要求的 ISL/OSL、起訖時間戳
├── metrics.csv      # 比較表
└── engine_metrics.csv        # 各工作點的服務端指標（有抓到的話）
```

server 節點上 `gpu_sample.sh` 的產出：

```
telemetry/
├── host.txt         # 靜態設定、GPU 型號、PCIe 連結，以及當下 GPU 上還有誰的 process
├── gpu.csv          # 使用率、記憶體、功耗、SM 時脈與上限、溫度、throttle 旗標
├── cpu.csv          # 全核與最忙單核的忙碌率、iowait
├── pcie.csv         # 每張卡的 rx/tx MB/s（nvidia-smi dmon 可用時）
├── nvlink.csv       # NVLink 累計計數器（只有 NVLink 板卡才有）
├── ib.csv           # 每個 IB port 的位元組數與 xmit_wait（只有 IB/RoCE 主機才有）
└── dcgm.csv         # DCGM profiling 計數器（有跑 exporter 時）
```

`metrics.csv` 的欄位，每個工作點一列：

`point`、`pass_rate`、`ttft_avg`、`ttft_p90`、`ttft_p99`、`itl_avg`、`e2e_avg`、`e2e_p90`、
`e2e_p99`、`out_tok_per_s`、`req_per_s`、`goodput`、`req_count`、`good_count`、`isl`、`osl`

- **goodput** —— 每秒有多少請求**同時滿足所有** SLA 門檻。這才是要最佳化的數字；
  原始吞吐量可能看起來很漂亮，實際上多數請求都爆了延遲預算。
- **pass_rate** —— `good_count / req_count`。只看 goodput 不看達標率會誤導。
- 延遲單位都是毫秒；`isl` / `osl` 是**實測**的序列長度，值得跟你要求的值對一下。

---

## 3. `gpu_sample.sh`

`metrics.csv` 告訴你 goodput 是**多少**；這支告訴你它**為什麼不再往上**。
在跑服務的那個節點上執行，`run.sh` 則從別的地方打流量：

```bash
# 在 server 節點上
./gpu_sample.sh --out ./telemetry --duration 900

# 事後，或針對已收集的目錄
./gpu_sample.sh --summary ./telemetry
```

| 旗標 | 預設 | 說明 |
| --- | --- | --- |
| `--out DIR` | 必填 | 輸出 `host.txt`、`gpu.csv`、`cpu.csv`，硬體支援時再加上 `pcie.csv` / `nvlink.csv`。 |
| `--summary DIR` | — | 重印既有目錄的摘要，不做取樣。 |
| `--interval SEC` | `1` | 取樣週期，可用小數。 |
| `--duration SEC` | 直到 Ctrl-C | 跑滿指定秒數後自動結束並印摘要。 |
| `--label NAME` | 無 | 記進 `host.txt`。 |
| `--ib-root DIR` | `/sys/class/infiniband` | 從哪裡讀 IB port 計數器。主要是為了對假的目錄樹做測試。 |
| `--dcgm-url URL` | `http://localhost:9400/metrics` | 站台有跑 DCGM exporter 的話。只探測一次，沒有也無妨。 |
| `--no-dcgm` | 開啟探測 | 完全跳過。 |

只需要 `nvidia-smi`、`/proc/stat` 與 sysfs —— 不用 DCGM，不用 root，也不需要 rdma 工具鏈。
它會記錄每張卡的使用率、記憶體用量、**功耗相對於板卡上限**、**SM 時脈相對於該卡最大時脈**、
溫度、依嚴重程度分類的 throttle 原因、**PCIe rx/tx 吞吐量**、有 NVLink 的話還有 NVLink 計數器、
**InfiniBand 每個 port 的吞吐量與壅塞**、全核與最忙單核的 CPU 忙碌率，
以及取樣開始與結束時**每張 GPU 上所有的 compute process**。

摘要怎麼讀：

| 訊號 | 結論 |
| --- | --- |
| 功耗接近上限、時脈滿檔 | GPU 真的被用滿了 —— 想再快只能改模型／引擎，加流量沒用。 |
| 功耗遠低於上限 | GPU 在**挨餓**：batch 太小、被 client 卡住，或請求在排隊。 |
| SM 時脈低於最大值的 85% | 真的在降頻 —— 這組數字不可重現，先解決散熱／供電再談調校。 |
| throttle 旗標亮著**但**時脈滿檔 | 忽略它。見下。 |
| 記憶體峰值遠低於總量 | KV cache 給太少；調高 `gpu_memory_utilization`。 |
| PCIe 峰值接近連結容量 | 互連瓶頸 —— TP 走 PCIe、或 PD 分離架構搬 KV 時就會撞到。 |
| IB 接近連結容量 | 跨節點路徑就是天花板。對 PD 分離而言，這會限制 P:D 比例與 worker 擺放。 |
| IB 離容量還遠但 `xmit_wait` 在漲 | 是壅塞或 credit 不足，不是頻寬。該看繞送與流量樣態，不是連結速度。 |
| 單核滿載、平均很低 | 是序列化階段（HTTP 前端、detokenize）卡住，不是 GPU。 |

**throttle 旗標不是證據。** `SwPowerCap` 與 `SwThermalSlowdown` 在很多卡上是長期亮著的，
而 SM 照樣跑滿速 —— 在開發這份工具的機器上，旗標在約 85% 的取樣裡都亮著，時脈卻穩定在
最大值的 95%。所以取樣器會把 `clocks.max.sm` 跟 `clocks.sm` 一起記下來，並把旗標拆成
硬體（真的掉時脈）與軟體（僅供參考）兩類，摘要與 `diagnose.sh` 都改以**時脈比例**判定。
只看旗標的規則會讓每個工作點永遠印 `THROTTLING`，把真正的瓶頸整個蓋掉。

**站台有 DCGM exporter 的話一定要用。** 它是這裡唯一「量測」而不是「推論」的來源：

| 欄位 | 它能定案什麼 |
| --- | --- |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor core 到底忙不忙。高就是算力瓶頸，不需要拿功耗來論證。 |
| `DCGM_FI_PROF_DRAM_ACTIVE` | 記憶體介面，實測而非推論 —— 也是對 roofline 的獨立檢查。 |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE`、`SM_ACTIVE` | 真正的佔用率，而 GPU-Util% 不是。 |

Exporter 就是一個 Prometheus 端點，所以這裡照樣輪詢它並寫出 `dcgm.csv`。沒有它一切照常：
`dram` 欄位會退回 `utilization.memory`，而且圖例會標明來源。

對 PD 部署來說，tensor 數字還能檢查切法本身 ——
一個 tensor 不忙的 prefill 節點、或一個 tensor 很忙的 decode 節點，都不是你以為的那件事。

**`nvidia-smi` 的 GPU-Util% 不是飽和度指標** —— 它只代表取樣窗內至少有一個 kernel 在跑，
所以 LLM decode 會顯示 95–100%，但 SM 其實大半閒著。功耗相對板卡上限是比較好的替代指標；
真正的佔用率要靠 DCGM（`DCGM_FI_PROF_SM_ACTIVE`、`PIPE_TENSOR_ACTIVE`、`DRAM_ACTIVE`），
而這支腳本刻意不依賴它 —— `diagnose.sh` 的 roofline 改從模型形狀回答同一個問題。

**這裡每個數字都是整機的。** 在共用節點上，別人的 job 就混在你的功耗、時脈與 CPU 裡。
因此 `host.txt` 會在取樣開始**與結束**時各記一次所有 compute process（uuid、pid、使用者、
記憶體）—— 中途才進來的 job 只會污染後半段，而你看得出來。`diagnose.sh` 會把「有第二個
使用者」轉成明確警告：`GPU COMPUTE` 與 `SERIAL CPU` 這兩個判定未經證實。

**PCIe 吞吐量來自 `nvidia-smi dmon`**，它有自己的取樣迴圈；每一行在抵達時就蓋上 epoch
時間戳，讓 `pcie.csv` 跟 `gpu.csv` 走同一個時鐘。NVLink 計數器是累計值，在摘要時才換算成速率。

**InfiniBand 直接讀 sysfs** —— `/sys/class/infiniband/<dev>/ports/<port>/` ——
所以完全不需要安裝 rdma-core 那套工具。有三個細節很重要：

- `port_rcv_data` 與 `port_xmit_data` 的單位是**四個 octet**，所以取樣器會乘以四，
  `ib.csv` 裡是位元組。這點弄錯會把整個 fabric 低估 4 倍。
- 連結容量取自 port 自己的 `rate` 檔（`400 Gb/sec (4X NDR)` → 50 GB/s），
  所以百分比是對照它**實際協商到**的速率，不是它應該要有的速率。
  非 `ACTIVE` 的 port 會被排除，避免一條斷掉的連結稀釋掉容量。
- `port_xmit_wait` 計算的是「有資料要送但沒有 credit」的 tick 數。
  它能把**壅塞**跟**頻寬**分開：一個只用到 40% 線速但 `xmit_wait` 一直漲的 fabric
  並不缺頻寬，換更快的連結也救不了它。

錯誤計數器（`link_downed`、`symbol_error`、`port_rcv_errors` 等）是記開始與結束兩次，
而不是持續取樣 —— 一條在執行中途重新訓練過的連結，會讓它前後量到的東西全部不可信，
摘要會直接說出來。

---

## 4. `diagnose.sh`

其他腳本負責產生證據，這支負責下判斷。

```bash
./diagnose.sh ./aiperf-20260817-143022
./diagnose.sh ./aiperf-20260817-143022 --telemetry ./telemetry   # 證據強很多
./diagnose.sh ./run --telemetry ./tel --tp 2 --mem-bw 3350       # 手動指定 roofline 輸入
```

| 旗標 | 預設 | 說明 |
| --- | --- | --- |
| `--telemetry DIR` | 無 | `gpu_sample.sh` 的輸出目錄。多數判定都靠它。 |
| `--weights GIB` | 從模型目錄加總 | 整個模型的權重大小。 |
| `--tp N` | server 節點上的 GPU 數 | Tensor parallel 度數；權重與 KV 都按它切分。 |
| `--mem-bw GBPS` | 依 GPU 型號查表 | **每張卡**的 HBM 峰值頻寬。 |
| `--model-dir DIR` | `env.txt` 裡的 `tokenizer` 路徑 | 去哪裡找 `config.json` 與權重檔。 |

```
 point      goodput   pass     kv  queue preempt  power   clk  membw    mbu   vram   pcie  verdict
 c8            3.90   1.00    12%      0       0    40%    ok    70%    59%    67%     3%  HEADROOM
 c32          12.08   1.00    47%      0       0    60%    ok    70%    77%    67%     3%  MEMORY BW
 c64          17.40   0.97    83%      2       0    86%    ok    70%    84%    67%     3%  MEMORY BW
 c128         12.81   0.61    99%     41     292    94%    ok    70%    70%    67%     3%  KV CACHE

 peak goodput  17.40 req/s at c64 (pass rate 0.972)
 bottleneck    MEMORY BW -- MBU 84% -- the memory system is moving the weights + KV per
               decode step at 84% of peak bandwidth; more SMs cannot help
 memory        KV cache hit 99% while only 67% of VRAM was in use
               -> raise gpu_memory_utilization; you are leaving 28% of the card unused
 SLA missed    at c128: TTFT p90 2400ms > 1000ms; e2e p90 7600ms > 5000ms
 SLA cliff     pass rate first drops below 0.99 at c64 (0.972)
 repeatability anchor vs c32: 1.5% apart (stable)
```

**為什麼要工具而不是用眼睛看：** 每個來源單獨看都有歧義。GPU 功耗低這件事，
對「client 打不動」「排程器上限」「序列化 CPU 階段」三種情況都成立
—— 只有**組合起來**才分得開。判定依結論強度排序：

| 判定 | 證據 | 處理 |
| --- | --- | --- |
| `THROTTLING` | SM 時脈低於最大值 85% 的取樣超過 20%、**且功耗達板卡上限 30% 以上**，或硬體降頻超過 5% | 撞到功耗／溫度牆 —— 數字不可重現，先修這個再談調校。（**閒置**的 GPU 也會降頻，那不是 throttling，所以需要功耗下限這個條件。） |
| `KV CACHE` | preemption > 0 且 cache 接近滿 | 超賣了。降並行數，或加大 cache 容量。 |
| `MEMORY BW` | **MBU ≥ 60%** **且** KV 佔傳輸量 ≥ 50% | 真的是頻寬瓶頸。加大 batch 只會等比增加 KV 傳輸量，唯一的槓桿是減少每 token 的位元組數（量化、推測解碼、縮短 context）。 |
| `WEIGHT BOUND` | **MBU ≥ 60%** 但 KV 只佔一小部分 —— 頻寬是花在為小 batch 重讀權重 | 這不是頻寬牆。權重讀取是每個 decode step 的**固定**成本，加大 batch 就能攤掉。先把並行數推上去，再考慮動模型。 |
| `INTERCONNECT` | PCIe **或 InfiniBand** 峰值 ≥ 連結容量 70%，而 GPU 功耗仍有餘裕 | TP 或 PD 分離的傳輸是天花板；判定會指名是哪一條 fabric。看 `nvidia-smi topo -m`、P:D 比例與 worker 擺放。 |
| `GPU COMPUTE` | **DCGM tensor pipe 活躍 ≥ 50% 的週期**，或功耗 ≥ 板卡上限 85% 且 MBU 低於 60% | 真的是算力吃滿 —— 想再快只能改模型／引擎。tensor 計數器是直接證據，功耗只是它的代理；一個 prefill 節點可能在 59% 功耗下就已經是算力瓶頸。 |
| `MEMORY BW?` | 記憶體控制器忙碌 ≥ 70%、功耗偏低，**且沒有 roofline 輸入** | 這是代理指標，不是結論。補 `--weights` / `--mem-bw` 才能定案。 |
| `SCHEDULER CAP` | 佇列很深、KV 很低、GPU 有餘裕；或 MBU 很高但幾乎都是權重讀取而請求還在排隊 | 調高 `max_num_seqs` / `max_num_batched_tokens`。 |
| `SERIAL CPU` | 單核滿載、GPU 閒置 | HTTP 前端或 detokenize 是天花板。 |
| `CLIENT BOUND` | 服務端實際併發遠低於要求、佇列是空的 | 壓測端跟不上 —— 把它移出 server 節點。 |
| `HEADROOM` | SLA 有達標、沒有排隊 | 並行數還可以往上推。 |

### Roofline：算力還是頻寬，用算的

「記憶體頻寬瓶頸」是 LLM 服務裡最常見的說法，也是最難在沒有 profiler 的情況下驗證的一句。
`diagnose.sh` 不用猜功耗，直接算：

```
每個 decode step 每張卡要搬的位元組 = (權重 + batch × 每條序列的 KV) / tp
MBU = 每步位元組 × 每秒步數 / HBM 峰值頻寬
```

`每秒步數` 就是 `1000 / itl_avg`，直接來自 `metrics.csv`。其餘都來自模型與卡本身，
全部自動偵測、也全部可以覆寫：

| 輸入 | 自動來源 |
| --- | --- |
| 權重大小 | 模型目錄裡 `*.safetensors` 的檔案大小總和（`env.txt` 的 `tokenizer` 路徑） |
| 每 token context 的 KV 位元組 | `config.json`：`num_hidden_layers`、`num_key_value_heads`、`head_dim`、`torch_dtype` |
| HBM 峰值頻寬 | telemetry `host.txt` 裡記錄的 GPU 型號 |
| tensor parallel 度數 | server 節點上的 GPU 數量 |

**MBU ≥ 60%** 代表記憶體系統就是天花板，加算力沒用。**MBU 遠低於 60% 但功耗很高**則相反
—— 那真的是算力瓶頸。

但記憶體系統被吃滿，**取決於裡面在搬什麼**，意義完全不同，所以報告會依每步位元組中的
KV 佔比再分一次：

| 傳輸組成 | 判定 | 為什麼重要 |
| --- | --- | --- |
| KV 佔 ≥ 50% | `MEMORY BW` | 加大 batch 會等比增加 KV 傳輸，幫不上忙。要減少每 token 的位元組數。 |
| KV 佔一小部分，主要是權重讀取 | `WEIGHT BOUND`；若請求正在排隊則是 `SCHEDULER CAP` | 權重讀取是每個 decode step 的**固定**成本，加大 batch 就能攤掉。該做的是提高並行數（或 `max_num_seqs`），不是換小模型。 |

只看 MBU 會把這兩種都叫做「頻寬瓶頸」，然後叫你去量化一個真正問題只是 batch 只有 8 的模型。

這個差別從外面完全看不出來，而且正是舊版判斷錯的地方。上面那輪裡 `c64` 的功耗是板卡上限的
86%，只看功耗的規則會判 `GPU COMPUTE`「真的用滿了」；roofline 顯示 MBU 是 84%
—— 那些電力是花在搬權重和 KV，換一張 FLOPs 更高的卡不會有任何差別。

### 拿硬體檢查 roofline

站台有跑 DCGM exporter 時，MBU 就不再是唯一的答案。`DRAM_ACTIVE` 直接量測記憶體介面，
所以報告會把兩者並排：

```
 cross-check   roofline MBU 12% vs measured DRAM_ACTIVE 77% of cycles
               -> the hardware is moving far more than the model shape accounts
                  for. On an MoE this is the signature of a batch touching many
                  more experts than the active-parameter count assumes -- re-run
                  with --weights set to the whole checkpoint for the upper bound.
```

兩者不是同一個量 —— `DRAM_ACTIVE` 數的是介面忙碌的週期數，MBU 數的是位元組對峰值的比例，
所以 `DRAM_ACTIVE` 正常會等於或高於 MBU。**帶資訊的是分歧的大小**，而且兩個方向都有意義：

| 差距 | 解讀 |
| --- | --- |
| MBU 高出 `DRAM_ACTIVE` 25 個百分點以上 | roofline 宣稱的記憶體用量超過硬體實際看到的。檢查 `--weights`、`--tp`、`--mem-bw`。 |
| `DRAM_ACTIVE` 高出 MBU 35 個百分點以上 | 硬體搬的比模型形狀能解釋的多 —— 在 MoE 上，就是 batch 碰到的 expert 遠多於 active 參數量的假設。 |
| 中間 | 一致。頻寬這個讀數有兩個獨立來源互相佐證。 |

**中間那一列正是這題的關鍵。** 沒有它，MoE 的問題只有「用 active 權重跑一次、用完整
checkpoint 跑一次，真值在中間」這種兩端答案。`DRAM_ACTIVE` 會告訴你實際落在哪一端。

表頭還會印出 **ITL 下限**：在這個模型大小與頻寬下，單 token 延遲最快能到多少。
實測 ITL 如果已經貼著它，再怎麼調引擎都不會更快。MBU 超過 100% 是物理上不可能的，
代表某個輸入錯了，報告會直接說並指出最可能的兇手（通常是 `--tp`）。

沒有 `--telemetry` 時，只要自己給 `--weights` 與 `--mem-bw`，roofline 一樣算得出來
—— 它需要的是模型與卡的規格，不是取樣器。

**記憶體是兩個不同的問題**，報告刻意分開呈現：

| 欄位 | 意義 | 說明 |
| --- | --- | --- |
| `mbu` | 實際達成頻寬 ÷ HBM 峰值頻寬 | **頻寬，實測。** 判定算力 vs 頻寬就看這欄。 |
| `membw` | `utilization.memory` —— 記憶體控制器忙碌時間佔比 | 同一件事的**代理指標**，只在缺 roofline 輸入時才用。 |
| `vram` | `memory.used / memory.total` | **容量。** 這張卡實際被佔用了多少。 |
| `kv` | 引擎自報的 KV block 使用率 | **引擎拿到的那塊**用掉了多少。 |

`kv` 與 `vram` 這一對專門抓浪費掉的硬體：KV cache 到 96% 而 VRAM 只有 60%，
就代表 `gpu_memory_utilization` 設太低、整張卡有三分之一在閒置 —— 通常是最便宜的
goodput 提升。報告會明講；反過來 VRAM 峰值超過 97% 時也會警告（有 OOM 風險）。

### 多個 worker，一個數字

PD 部署是**每個 worker 各有一份指標**，而把它們併起來需要依指標種類用不同規則。
把 cache usage 這種**分數**在六個 worker 上相加會得到 600%；
把佇列深度取最大值則會藏掉整隊真正的積壓。所以 `observe.sh` 把各端點分開存，
`diagnose.sh` 再依種類合併：

| 欄位 | 跨 worker 的合併方式 | 為什麼 |
| --- | --- | --- |
| `kv` | 取最大 | 這是分數。有一個 worker 到 99% 就是瓶頸，其他人如何無關。 |
| `queue` | 取總和 | 積壓只有以整隊來看才有意義。 |
| `preempt` | 取總和 | 這是事件次數。 |
| roofline 的 batch | **取最大**，不是總和 | 每個 worker 都要重讀**自己那份**權重，所以決定 MBU 的是單一 worker 的 batch。取總和會高估，倍數就是 worker 數。 |
| `CLIENT BOUND` 判定 | 取總和 | 這一條問的確實是「所有地方總共有多少請求在飛」。 |

### 擾動：換個輸入，這個結論還站得住嗎？

只要這輪有 `--isl-sweep` / `--osl-sweep` 的點，報告就會多一段 —— 不是重述結論，而是去測試它：

```
 perturbation -- one input varied at a time at c=32

   ISL sweep (OSL fixed): prefill work and KV context both grow
          isl   ttft_avg    itl_avg   prefill tok/s
         1000        115      11.88            8696
         2000        230      14.16            8696
         4000        460      18.73            8696
         8000        920      27.88            8696
   . prefill throughput held within 0% across a 8.0x ISL range -- TTFT is linear
   . ITL slope   measured 2.29 us per token of context
                   roofline predicts 2.06 us at c=32
     -> within 11%: KV traffic explains how ITL grows, so the bandwidth
        model is confirmed by perturbation, not just inferred from power

   OSL sweep (ISL fixed): decode steps grow, nothing else does
          osl   ttft_avg    itl_avg    e2e_avg
           50        460      18.56       1370
          200        460      18.73       4188
          800        460      19.42      15975
   . ITL moved +5% across a 16.0x OSL range -- each decode step costs the same,
     which is what a steady-state decode bottleneck looks like
```

中間那段才是整件事的重點。roofline 只憑模型形狀就能預測「context 每多一個 token，
ITL 應該多多少」——`batch × 每 token KV / (tp × 頻寬)`，完全不參考任何量測值。
而 ISL 掃描則把同一個斜率實際量出來。預測值會先除以實測的 MBU —— 因為跑在峰值 90% 的引擎，
斜率本來就是理想值的 1/0.9 倍；把它拿掉之後，剩下的才是純粹對「context 放大」的測試，
而不是在測引擎效率。兩者吻合時，`MEMORY BW` 這個判定就是被**測試過**的，
不是推測出來的，解讀也就從相關性升級成因果。

兩者不吻合時，報告會說是往哪個方向差，以及那代表什麼：

| 實測 vs 預測 | 解讀 |
| --- | --- |
| 差距在 ~30% 以內 | KV 傳輸量足以解釋 ITL 的成長 —— 頻寬模型成立。 |
| 陡很多 | 有別的東西隨 context 放大：attention kernel、排程器、cache 抖動。不是頻寬。 |
| 平很多 | 引擎搬的 KV 比假設少 —— prefix caching、量化過的 KV cache，或 `--tp` 給錯了。 |
| ITL 對 ISL 完全不變 | KV 項根本不適用；回頭檢查 `--tp` 與 KV 的 dtype。 |

另外兩項檢查則跟 roofline 無關：

- **ISL 掃描的 prefill 成本** —— TTFT 會被擬合成 `固定開銷 + 邊際成本 × ISL`,
  線性與否是看邊際成本沿著掃描有沒有變。（直覺的 `ISL / TTFT` 反而會誤導：
  固定開銷大的時候它會隨 ISL **上升**，完全不能拿來判斷線性。）
  短端由固定開銷主導時報告會直說 —— 那裡的短 prompt 量到的是壓測框架，不是 prefill。
- **OSL 掃描的 ITL** —— 應該幾乎不動，因為只有 decode 步數改變。
  漂移很大代表每步成本不是固定的，通常原因是 KV cache 在長大：那是容量壓力，不是穩態。

### 它不會讓你相信的事

它同時會標出那些會悄悄讓整輪失效的狀況：

- 實測 ISL/OSL 與要求值偏離超過 5%
- goodput 在測到的最大並行數還在往上（你的 sweep 根本沒走到峰值）
- 對齊點與 sweep 同並行數差異超過 5%（量測本身雜訊太大，小差異不可信）
- **兩台機器時鐘不一致** —— telemetry 完全沒跟這輪重疊時，會直接印出差了幾秒
  以及怎麼修正，而不是安靜地給你一整排空欄位
- **server GPU 上有第二個使用者的 job**，這會讓整機層級的 GPU 與 CPU 判定失去證明力
- **GPU 根本沒醒過來** —— 每個工作點的功耗峰值都低於板卡上限 15%，代表取樣器跑錯節點、
  或工作根本不在那些卡上；此時所有 GPU 欄位都沒有意義，報告會直說而不是照樣排名
- **工作點短於五個輪詢週期**，此時佇列、KV、running 都被欠採樣，peak 會低估真實值
- **亮著但沒有造成掉時脈的軟體 throttle 旗標**，會明確說「已忽略」，免得你去追一個假訊號

只有 `metrics.csv` 也能跑，給得越多說得越多 —— `engine_metrics.csv` 補上佇列／cache／
preemption 欄位，`--telemetry` 補上功耗、時脈、互連與 CPU，
而 roofline 的輸入則把頻寬這題從猜測變成一個數字。

---

## 5. `official.sh` —— 計分指令，一字不差

題目規則把官方計分重跑的 AIPerf 指令**鎖死**了，改動它分數就無效。這支腳本裡就只有那個指令，
沒有監控、沒有便利功能、沒有任何額外旗標。

```bash
./official.sh --host gw01:8000 --model qwen3-pd --concurrency 64
./official.sh --print --host gw01:8000 --model qwen3-pd --concurrency 64   # 只印不跑
./official.sh --verify                                                     # 跟規則檔對照
```

| 旗標 | 預設 | 說明 |
| --- | --- | --- |
| `--concurrency N` | 必填 | 團隊宣告的 N，同時成為 `--concurrency` 與 `--workers-max`。 |
| `--host HOST:PORT` | 必填 | 團隊的 gateway。只收 `host:port` 不收 URL —— 指令自己會組 `http://$HOST`，給 URL 會被擋下。 |
| `--model ID` | 必填 | gateway 在 `/v1/models` 回報的 id。 |
| `--out DIR` | `./official-<時間戳>-c<N>` | 腳本會**進到**這個目錄再執行。 |
| `--venv PATH` | `./aiperf-venv`，否則腳本旁邊那個 | 兩種都找，所以 `observe.sh` 從點目錄呼叫它也能解析到。 |
| `--print` | — | 印出確切指令後結束。 |
| `--verify [FILE]` | `../2026-APAC-HPC-AI/1. Qwen3 AI Task Rules.md` | 跟規則對照，有任何差異就回非零。 |

**`--verify` 才是這支檔案存在的理由。** 「我們跑的就是官方指令」是一句宣稱，這讓它變成可驗證的事實。
它會從主辦方的 markdown 重新抽出那段指令、兩邊都 tokenize、逐一比對 —— 打錯字、拿到舊版、
或比賽中途規則被更新，都會大聲失敗，而不是安靜地讓一次計分重跑作廢：

```
x THE COMMAND HAS DIVERGED FROM THE RULES
   --- rules
   +++ official.sh
    --benchmark-duration
   -180
   +120

   the rules are authoritative: edit official.sh to match, never the reverse
```

**公布的指令裡沒有 `--artifact-dir`**，加上去就是改動它。所以腳本改成 `cd` 進 `--out`，
讓 AIPerf 的預設輸出目錄（`artifacts`）自然落在那裡。AIPerf 還會再往下開一層
`artifacts/<model>-<endpoint>-concurrency<N>/` —— 這也是為什麼 `observe.sh` 是用**搜尋**的，
而不是假設路徑。

**只有三個值會被代入**，就是規則留給團隊決定的那三個。其餘全部固定：tokenizer id、
seed 2026、4000/200 的形狀、`--cache-bust first_turn_prefix`、closed-loop 模式、goodput 門檻。

---

## 6. `observe.sh` —— 監控放在外面

計分指令不能帶監控旗標，所以監控只能發生在它**周圍**。`observe.sh` 把任何指令當成子行程執行，
在完全不碰它的前提下收集證據：

```bash
# 每個 server 節點上，跑滿整段時間
./gpu_sample.sh --out ./tel

# client 端
./observe.sh --out ./scored --gateway gw01:8000 -- \
    ./official.sh --host gw01:8000 --model qwen3-pd --concurrency 64

./diagnose.sh ./scored --telemetry ./tel --weights 22 --tp 8
```

| 旗標 | 預設 | 說明 |
| --- | --- | --- |
| `--out DIR` | 必填 | 點會累積在這裡；每個宣告並行數呼叫一次就湊成一條曲線。 |
| `--point NAME` | `c<N>` | `diagnose.sh` 把 `c<N>` 的點視為並行數曲線。 |
| `--concurrency N` | 從子指令裡讀 | 決定點的名稱，也是 roofline 的 batch size。 |
| `--gateway HOST:PORT` | 無 | 啟用 model id 查詢、時鐘偏移量測，以及預設的 metrics URL。 |
| `--metrics-url URL` | `http://<gateway>/metrics` | Prometheus 端點。**可重複 —— 每個 worker 給一次。** |
| `--metrics-interval` | `2` | 輪詢週期（秒）。 |
| `--isl` / `--osl` | `4000` / `200` | 記進 env.txt 給 roofline 用；預設值就是題目的形狀。 |
| `--sla` | `"time_to_first_token:3000 inter_token_latency:20"` | 記下來讓 `diagnose.sh` 用真正的閘門判定。 |
| `--no-metrics` | 開啟 | 完全不輪詢。 |

它收集三樣指令本身做不到的東西：

- **執行期間持續輪詢的 Prometheus 指標。** 佇列深度與 KV 使用率都是 gauge：負載一停就歸零，
  所以前後各抓一次等於什麼都沒抓到。
- **跟 gateway 的時鐘偏移**，取自服務端自己的 HTTP `Date` 標頭，
  讓 server 節點上取樣的 telemetry 能對齊執行窗。
- **每個點的起訖時間戳**，那正是能做上面那個切片的前提。

接著它會去找子行程產出的結果（不管放在哪），組出 `metrics.csv`、`engine_metrics.csv`、
`points.csv` 與 `env.txt` —— 剛好就是 `diagnose.sh` 要吃的那一組。同一個點重跑會**取代**該列，
不會變成兩列。

它也會直接印出這題真正計分的數字：

```
  c64: goodput 27.49 req/s, pass rate 1.000  ->  880 good output tok/s
```

---

## Qwen3-235B-A22B PD 題目的注意事項

### `build/` —— 重建環境

提交要求附上「能讓別人重建出這次量測所在環境」的建置方法。這件事有兩個誠實的答案，
`build/build.sh` 兩個都支援：

```bash
./build/build.sh                 # 依 build/Dockerfile* 建置並匯出 .sqsh
./build/build.sh --record-only   # 你直接用叢集上提供的映像：把它們的指紋記下來
```

如果你原封不動用了主辦方提供的 `sglang-harbor.sqsh`，就用 `--record-only`。
它會記下你**實際跑的那份映像**的大小與 SHA-256，那是可查證的宣稱；
附一個你根本沒拿來建置的 Dockerfile 則不是。

| 檔案 | 釘住什麼 |
| --- | --- |
| `build/Dockerfile` | SGLang 服務堆疊，**用 digest 釘住**，外加 PD 傳輸後端要的 NIXL。 |
| `build/Dockerfile.aiperf` | 客戶端：釘住版本的 AIPerf，以及預先烘進去的官方 tokenizer。 |

兩者都刻意用 digest 或精確版本釘住。浮動 tag 等於「我們量測的堆疊」是登錄檔那天剛好放著的東西，
既不可重現、在報告裡也站不住腳。兩者都會把 `pip freeze` 的清單寫進映像，
之後的跑法可以直接跟產生某個數字的那次做 diff。

有三個決定是**真的動手建置之後**才確定的：

- **`UCX_NET_DEVICES` 刻意不烘進映像。** 正確的 HCA 每個節點不同，所以在啟動時才偵測。
  烘進去正是「整個 fabric 的 KV 流量全擠在一個 port」的經典成因。
- **客戶端預先抓 tokenizer，而且在建置時就用離線模式證明它可用。**
  官方指令傳的 `--tokenizer Qwen/Qwen3-235B-A22B-Instruct-2507` 是 HF id，
  沒有對外網路的節點會在**你已經花掉配額之後**才失敗。把快取烘進去完全不改動那個指令。
  接著建置階段會關掉網路載入一次，不成功就讓 build 失敗 ——
  因為一個缺了 `config.json` 的快取看起來是滿的，卻仍然會去連網，而這正是第一版的下場。
- **裝 `nixl-cu12` 而不是 `nixl`。** 裸的 meta-package 會同時依賴 cu12 與 cu13 兩個變體，
  而且各自都會拉一份 torch。

### 先跑煙霧測試 —— `run/01`、`02`、`03`

Application notes 裡三個逐級加深的檢查，每個都是單獨一次提交，**結束碼就是全部的判定**。
在把配額花在 PD 拓撲上之前先跑完它們：

| 腳本 | 證明什麼 |
| --- | --- |
| `01-smoke-image.sbatch` | 容器起得來、容器內看得到 GPU、`sglang` import 得動、`$HOME` 可寫。 |
| `02-smoke-serve.sbatch` | 單一非分離的 server 載得動模型、變成 healthy，並回報它會回應哪個 model id。 |
| `03-smoke-serve-aiperf.sbatch` | 客戶端映像、網路路徑與 server 能完成一次真正的串流往返。 |

```bash
sbatch run/01-smoke-image.sbatch
SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/02-smoke-serve.sbatch
SMOKE_MODEL=/scratch/models/Qwen3-0.6B sbatch run/03-smoke-serve-aiperf.sbatch
```

**2 和 3 請用小模型。** 它們測的是模型載入與服務路徑，0.6B 一分鐘就給得出答案，
而用 235B 得花掉一大塊配額才學到同一件事。**而且要用 sbatch 提交** ——
不要用 `srun --pty`。互動式 session 只要開著就一直佔住 GPU 配額，
而依近期用量調降優先度的排程器會因此把你自己之後的提交往後推。

Step 3 的結束碼是 0 並不等於通過：它還會檢查「真的有量到請求」，
因為 AIPerf 可能乾淨地結束卻什麼有用的東西都沒寫出來。

### `run/sweep-pd.sbatch` —— 搜尋 Prefill:Decode 比例

切法是規則留給你的最大槓桿，而 application notes 那個「已驗證」的 8+8 是起點不是答案。
這支腳本在它周圍搜尋：

```bash
sbatch run/sweep-pd.sbatch
SPECS="2x4:4x2 2x6:4x1" CONCURRENCIES="64 128" sbatch run/sweep-pd.sbatch
```

一個拓撲就是一個 spec，格式 `<prefill-tp>x<數量>:<decode-tp>x<數量>`：

| Spec | 切法 | 是什麼 |
| --- | --- | --- |
| `2x4:4x2` | 8 + 8 | Application notes 驗證過的那個。 |
| `2x6:4x1` | 12 + 4 | 偏 prefill；prefill worker 會溢到第二個節點。 |
| `2x2:4x3` | 4 + 12 | 偏 decode。 |
| `4x2:8x1` | 8 + 8 | 同樣的 GPU 切法，但 TP 更寬。 |

擺放方式是填滿一個節點再溢到下一個，而且**沒有任何 worker 會跨節點** ——
tensor parallel 走 InfiniBand 的代價遠大於調比例能賺回來的。
`run/pd-lib.sh` 同時被這支和計分 job 使用，所以贏家只要換一個變數就能拿去跑：

```
 winner   2x6:4x1 at c64 -- 41200 good output tok/s (pass 0.981)

 Take it to a scored run with:
   PD_SPEC=2x6:4x1 CONCURRENCY=64 sbatch run/submit.sbatch
```

如果你要自己重寫，有三個決定值得保留：

- **部署一次、量完所有並行數、再拆掉。** 載入 235B checkpoint 的成本遠大於 15 秒的量測；
  而且在單一固定 N 下比較不同比例本來就會誤導 —— 最佳 N 會隨比例移動。
- **量測窗縮短，其餘一模一樣。** 規則明確不限制你自己的測試，所以窗口是 15 秒而不是 120。
  但 4000/200 的形狀與 goodput 門檻完全照官方指令，這樣這裡的通過率才跟那邊意義相同。
- **差距小於 5% 不算結果。** 摘要會直說，而不是宣布一個短跑根本分辨不出來的贏家。

### `run/submit.sbatch` —— 部署與計分在同一個 job

題目要求「把服務堆疊拉起來」與「呼叫官方指令」必須是**同一個 job 的連續步驟**。
`../run/submit.sbatch` 就是那個 job，照 Firmus application notes 裡已驗證的 8+8 拓撲寫的：

| | 節點 0（head） | 節點 1（tail） |
| --- | --- | --- |
| 角色 | 4× prefill | 2× decode |
| 平行化 | `--tp-size 2` | `--tp-size 4 --ep-size 4` |
| Port | 30100–30103（bootstrap 9000–9003） | 30200–30201 |
| 另外 | router 在 8000 | `--max-running-requests 256` |

那就是 `PD_SPEC=2x4:4x2`，也是預設值。任何擺得下的 spec 都可以 —— 上面的掃描會告訴你用哪一個。

它的執行順序本身就是重點：先驗證官方指令仍與規則一致 → 起 workers → 輪詢每個 worker 的
`/health` → 起 router → 輪詢它 → 在**兩個**節點上啟動 `gpu_sample.sh` →
透過 `observe.sh` 跑計分指令 → 停止取樣 → 對每個節點各做一次診斷。

```bash
sbatch run/submit.sbatch
CONCURRENCY=128 sbatch run/submit.sbatch     # 所有設定值都能用環境變數覆寫
```

如果你要自己重寫，有兩個細節值得保留：

- **`--verify` 跑在所有事情之前。** 一個已經跟規則不一致的指令，應該在第一秒就失敗，
  而不是在耗掉一次配額與 210 秒量測之後。
- **`UCX_NET_DEVICES` 是逐節點偵測的**，不是寫死的。NIXL 需要正確的 HCA 才能做跨節點
  KV 傳輸，而兩個節點列舉 device 的方式不保證一樣。找不到 ACTIVE port 時它會大聲說出來，
  不會讓 UCX 默默亂挑 —— 而不論哪種情況，`ib` 欄位都會顯示後果。

客戶端對話的 gateway 只能是 router；把壓測指向 prefill 或 decode worker 是違規。

### 另外四件會改變工具用法的事

**閘門是逐請求判定的，而且是 per-token 的閘門。** `TTFT ≤ 3000 ms` 且**平均** `ITL ≤ 20 ms`，
再加上 `good_request_fraction ≥ 0.90`。`diagnose.sh` 認得 `--sla` 裡的 `inter_token_latency`，
而且是用 `itl_p90` 而不是 `itl_avg` 判定 —— 因為閘門逐請求套用，聚合平均可以舒舒服服待在門檻內，
同時三分之一的請求已經掛掉：

```
 c64  itl_avg 18.49   itl_p90 21.64   pass_rate 0.562
 SLA missed    at c64: ITL p90 21.6ms > 20ms
```

當所有聚合值都過關但通過率沒過，報告會直說，並把你指向百分位，而不是假裝沒事。

**計分的是 good output token/s**，不是 good request/s。`diagnose.sh` 會把它
（`goodput × OSL`）跟排序用的 goodput 一起印出來。

**Qwen3-235B-A22B 是 MoE，roofline 的權重自動偵測對它是錯的。** 把 checkpoint 加總會得到約
235 B 參數，但每個 token 只啟用約 22 B。不處理的話 MBU 會高估約一個數量級，
所有點都會讀成 `MEMORY BW`。要明確給**啟用**的權重量：

```bash
./diagnose.sh ./scored --telemetry ./tel --weights 22 --tp 8   # FP8 下約 22 GiB active
```

這個數字背後藏著一個真實效應：batch 變大時不同請求會選到不同 expert，
每個 decode step 實際讀到的位元組會往完整 checkpoint 爬。這讓它變成一個**可測試的預測**
—— MBU 應該隨並行數上升，而且比單純 KV 項能解釋的更快 —— 而那正是並行數掃描與
`--isl-sweep` 存在的用途。

**InfiniBand 有被取樣，而且對這題來說它就是關鍵的那條鏈路。** PD 分離會把 KV cache
透過 8× NDR 在節點間搬運，題目自己的優化建議也點名了 cross-node KV transfer latency。
`gpu_sample.sh` 會從 sysfs 讀 port 計數器，`diagnose.sh` 以 `ib` 欄位呈現，
換算成協商到的連結速率的百分比：

```
 point      goodput   pass  ...   pcie     ib  verdict
 c32           9.52   1.00  ...     1%    90%  INTERCONNECT

 interconnect  InfiniBand peaked at 45.0 GB/s per port (90% of link), 180.0 GB/s across the node
               PD disaggregation moves the KV cache over this link -- at this
               level it is a real constraint on the P:D split and placement
 interconnect  IB xmit_wait grew by 15000: ports had data to send and no credit.
               That is congestion or credit starvation, not raw bandwidth
```

另外 `gpu_sample.sh` 是單節點的 —— **兩台都要跑**，然後各自診斷。

---

## 跟隊友對數字

數字要能並排，每個人必須一致的項目：

- 相同的 `--isl` / `--osl`
- 相同的 `--sla`（相同指標、相同門檻、相同單位）
- 相同的 `--anchor`，而且要單獨跑成一個點
- 暖機一律排除
- 釘住相同的 `aiperf` 版本
- 使用與服務端模型相符的本地 tokenizer
- 要比較擾動斜率的話，`--shape-c` 也要一致（ITL 斜率與 batch 成正比，
  不同並行數下量到的斜率沒有可比性）

只要旗標一致，`run.sh` 會把上面這些都處理好。有疑問時**先 diff 各自的 `env.txt`**
—— 不一致的地方都會在那裡現形。

---

## 疑難排解

| 症狀 | 原因／處理 |
| --- | --- |
| `$AIPERF not found` | 傳給 `run.sh` 的 `--venv` 跟安裝時不同，或根本沒跑過 `install.sh`。 |
| `could not get a model id from …/v1/models` | 端點沒有模型清單 —— 用 `--model` 明確指定。TRT-LLM 的 disagg proxy 請改問它其中一個 worker。 |
| `cannot recognize AIPerf's sequence length flags` | 遇到沒預期的 AIPerf 版本。看 `aiperf profile --help`，並更新 `run.sh` 裡的偵測區塊。 |
| `this version has no --streaming` | 版本太舊，量不到 TTFT/ITL。升級，或放棄這兩個指標。 |
| `no parsable results` | 每一輪都失敗了。往上讀錯誤訊息，並檢查 `warmup.log`。 |
| 某個掃描點沒有出現在表裡 | 那個並行數失敗了，但整輪依設計繼續跑；它的目錄裡沒有 `profile_export_aiperf.csv`。 |
| `metrics.csv` 一列都沒有 | 先確認 AIPerf 實際把結果寫到哪：`find $OUT -name profile_export_aiperf.csv`。有些版本會比組表程式找的位置再多一層。 |
| `diagnose.sh` 的 telemetry 欄位全是 `-` | telemetry 沒跟這輪的時間窗重疊。報告會印出差值與要加進 `env.txt` 的內容；通常是兩台機器沒對時。 |
| 每個點都判 `THROTTLING` | 在散熱受限的卡上這是真的：看表裡的 `clk` 欄與 `host.txt` 的溫度。時脈滿檔時不可能出現這個判定 —— 規則是看時脈，不是看旗標。 |
| 出現 `MEMORY BW?` 而不是 `MEMORY BW` | 缺 roofline 輸入。補 `--mem-bw`（模型目錄不是 tokenizer 路徑的話再補 `--weights`）。 |
| MoE 上每個點都讀成 `MEMORY BW` | 權重自動偵測把整個 checkpoint 加總了。用 `--weights` 給**啟用**的參數量。 |
| `official.sh --verify` 失敗 | 規則變了，或這份副本漂移了。規則是權威 —— 去改指令區塊來配合它。 |
| `official.sh` 找不到 venv | 給 `--venv`，或設 `$AIPERF_VENV`。它會先找 `./aiperf-venv` 再找腳本旁邊那個。 |
| `observe.sh` 說找不到 `profile_export_aiperf.csv` | 被觀測的指令失敗了，或寫到 `--out` 之外。它自己的輸出在 `<out>/<point>/`。 |
| 官方那一輪跑得比 210 秒久很多 | 前置不是免費的：ISL 4000 下 12800 筆合成對話約 220 MB，在 warmup 開始前要先花約 30 秒生成。 |

---

## 結束碼

兩支腳本只有在完全乾淨跑完時才回傳 0，所以可以直接當 CI 的檢查關卡。

| 腳本 | 什麼情況回非 0 |
| --- | --- |
| `install.sh` | Python／venv／pip 任一步失敗，或這個 AIPerf build 缺少 `run.sh` 依賴的旗標（結尾摘要會列出是哪些）。 |
| `run.sh` | 參數錯誤、前置條件不滿足、任何工作點失敗，或完全解析不到結果。 |
| `official.sh` | 參數錯誤、找不到 venv、AIPerf 回非零；或在 `--verify` 下與規則檔有任何差異。 |
| `observe.sh` | 被觀測指令回什麼就回什麼。監控本身失敗只降級成警告，絕不讓計分重跑失敗。 |

單一掃描點失敗**不會**中止整輪 —— `set -e` 是刻意不開的，才不會因為一個並行數壞掉就毀掉其餘的。
失敗的點會被記錄下來、列在結尾摘要，並反映在結束碼上。暖機失敗只會回報，不會讓整輪失敗。

---

## 注意事項與限制

- **整輪不是不可分割的。** 請確認 `metrics.csv` 裡該有的點都在；少一列就代表那個並行數
  沒有產出 `profile_export_aiperf.csv`。
- **原始產出檔很大。** `inputs.json` 與 `profile_export.jsonl` 記錄了每一個請求；
  進版控前先刪掉。真正值得留的是 `env.txt` 與 `metrics.csv`。
- **預設一輪是 5400 個請求**（暖機 40 + 對齊點 320 + 掃描 5040）。
  只想做煙霧測試的話，用 `--sweep` / `--req-per-c` 縮小。
  shape 掃描每個值再多加 `--shape-c × --req-per-c` 個；而且 `--osl-sweep` 的大值
  每個請求耗掉的實際時間，遠比請求數看起來的多。
- **因果驗證是選用的，而且只涵蓋兩個維度。** 不開 `--isl-sweep` / `--osl-sweep` 的話，
  並行數曲線只能把瓶頸跟負載關聯起來。開了之後，被擾動的也只有 ISL 與 OSL
  —— batch 組成、量化方式、平行化策略仍然只能靠跑不同輪再互相比較
  （用相同的 `--anchor`，然後 diff 各自的 `env.txt`）。
- **完全沒有取樣壓測端自己的資源。** `gpu_sample.sh` 是跑在 server 節點上的。
  `CLIENT BOUND` 這條判定依賴引擎回報 `num_requests_running`，所以對著沒有 Prometheus
  端點的服務，client 卡住是偵測不到的。壓測端請跑在核心數充裕的機器上，並自己盯它的 CPU。
- **prefill 與 decode 只能靠推論分開。** 證據只有 TTFT 與 ITL；
  服務端的 batch 組成（`num_batched_tokens`、iteration time）沒有被取樣。
- **NVLink 與 InfiniBand 的解析都沒有在真的那種硬體上驗證過。** 兩者都是在一台兩者皆無的
  機器上寫的 —— IB 是用一棵計數值已知的**合成 sysfs 樹**驗證的，兩者偵測不到時都會乾淨地
  退化成「沒有」。第一次在 SXM 或 IB 機器上跑時，請確認 `nvlink.csv` 與 `ib.csv` 的數字合理，
  並用 `perfquery` 或 `ibstat` 對照一次 IB 速率。
- **IB port 計數器會 wrap。** 舊版 sysfs 計數器在某些驅動上是 32-bit，在 NDR 速率下幾秒就會繞回。
  Wrap 會表現成負的差值；那些 port 會被排除在速率之外而不是報出錯誤數字，並且整輪會被標記。
  真的發生的話，就提高取樣頻率，或改讀 64-bit 的延伸計數器。
- **roofline 假設磁碟上的權重就是服務中的權重。** 載入時才量化（fp8、引擎自己做的 AWQ）
  會讓自動偵測的大小是錯的，那種情況請自己給 `--weights`。
