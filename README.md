# 2026 APAC HPC-AI · Qwen3-235B PD —— 跑過的每一支腳本

**這是 `run/` 和 `submissions/` 在 commit `dc9e1bd` 的快照。**
要修改請改 `../run/`，不要改這裡 —— 否則兩份會分歧。

## 從哪裡開始讀

| 先讀這個 | 為什麼 |
|---|---|
| **[`EXPERIMENTS.md`](EXPERIMENTS.md)** | **主線。21 個 trick 串成一條鏈**，每個有「單獨實驗？／影響／解讀／→ 下一步」。最後一欄就是下一列 |
| [`../handover/DIMENSIONS.md`](../handover/DIMENSIONS.md) | 橫向：12 個領域的每個維度，測過沒測過 ＋ 未測項目排序 |
| [`../handover/ERRATA.md`](../handover/ERRATA.md) | 被推翻的 10 個技術結論 ＋ 7 類方法錯誤 |
| [`../handover/CLUSTERS.md`](../handover/CLUSTERS.md) | 四個平台各自強制的規則 |
| [`../submissions/RUNS.md`](../submissions/RUNS.md) | 每一支 job 的完整記錄，700 行 |

## 這個資料夾的結構

| 目錄 | 內容 | 跑過的 job |
|---|---|---|
| `00-recon/` | 探查一個叢集能給什麼 | 每個叢集最早的那幾支 |
| `01-smoke/` | 五階煙霧測試：映像 → 服務 → 加客戶端 → TRT-LLM CLI → PD 配對 | 17604 等 |
| `02-campaign/` | **主力**：一個 allocation 掃完所有設定 ＋ 送出的那層 | NSCC 214749–216243、Nano5 372316–373046 |
| `03-probes/` | 剖面、DeepGEMM 預編譯、MoE 調校、記憶體探針 | 86188、215218、215916、216159、215960 |
| `04-nano5/` | Nano5 的建映像與四個實驗 | 372144、372289、372421、372456、373046 |
| `05-submissions/` | 六份提交（含 selftest），982 → 2,844 | — |
| `lib/` | 每一支都 source 的那些（排程器、叢集、引擎的 seam） | — |
| `tests/` | 24 支測試，**每一支對應一個真的失敗過的東西** | — |
| `AIPerf/` | 主辦單位的 kit，每個 job 都跑它 | — |

**每一支腳本旁邊有一個同名的 `.md`。** `lib/` 和 `tests/` 用目錄層級的表格
（22 和 25 個檔，一檔一份會是空殼）。

## 現在的分數

| 平台 | 引擎 | 最佳 | N | pass |
|---|---|---|---|---|
| **NSCC H100** | **SGLang** | **2,844** | 80 | 0.971 |
| Nano5 H100 | SGLang | 2,768 / 2,760 | 80 | 0.966 / 0.967 |
| Firmus H200 | SGLang | 3,591 | 80 | — |
| NSCC H100 | vLLM | 2,196 | 48 | — |
| NSCC H100 | 參考部署 | 982 | 24 | 0.580 |

兩個 H100 平台互相驗證，差 2.7–3.0%。**跨平台的 tok/s 不比大小**（規則第 73 行）。

## 怎麼重跑

```bash
cd /mnt/nfs/work/hsingyu/HPC-AI          # 用 run/，不是這份快照

# 掃一輪設定（主力）
PD_SITE=nano5 NODES=2 GPUS=8 ENGINES=sglang SPECS="4x2:4x2" \
  CONCURRENCIES="80 88" VARIANTS="bigpf:$BIG" \
  run/submit-site.sh run/10-campaign.sbatch

# 跑那個拿到 2,844 的獨立腳本
qsub submissions/05-nscc-h100-sglang-2844/qwen3-pd-h100-16gpu.pbs

# 送之前先驗（今天四次 0 SU 的失敗都是這樣抓到的）
bash -c '. run/sites/nano5.sh; nano5_ready'
for t in run/test-*.sh; do bash "$t"; done
```

## 五個有效的增益，全部在 prefill 或在關掉某個預設

| 改動 | 影響 | 性質 |
|---|---|---|
| 關掉 expert parallelism | **2.27×** | 它**預設是開的** |
| DeepGEMM 在 prefill | **+10.7%** | 只給一個 role（decode 上慢 1.3–1.5×） |
| 整數 GPU id | **+6.3%** | 修一個被吞掉的例外 |
| prefill batch 12,288 ＋ per-role mem_fraction | **+6.0% R** | 把浪費五倍的 KV pool 換成 DeepGEMM 的工作區 |
| router 改 `round_robin` | 救回一次失敗 | 預設 `cache_aware` 在這個 workload 上最差 |

**三個「看起來該有用」的全部否證**（都是配對實驗）：all-reduce 實作（2,712 對 2,710）、
symmetric memory（−3.4%）、MoE config 調校（兩次反向）。

**沒有一個增益來自「加更多東西」。**
