# Qwen3-235B PD 實驗對照表

本文件將每個實驗拆成兩張固定格式的表，方便直接橫向比較：

1. **設定表**：這支實驗跑了什麼、使用哪個腳本、改了哪些參數。
2. **結果表**：固定列出 good output tokens/s、pass、R、TTFT p90、ITL p90。

`—` 代表原始紀錄沒有量測該欄位；`不可讀` 代表有結果但受到 OOM、port collision
或其他執行問題影響，不能當成有效 benchmark 結果。`R` 是 request throughput
（req/s），`N` 是併發數。

## 腳本對應規則

NSCC 的主線 job 多數是在同一個 campaign driver 內以 cell／variant 執行，入口是
[`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)。因此表內若標成
「campaign driver」，表示實驗參數是由該腳本的環境變數或 `VARIANTS` 傳入，不代表
每個 job 都有一份獨立的 `.sbatch`。

| 類型 | 腳本 |
|---|---|
| NSCC／Nano5 campaign | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch) |
| MoE config tuning | [`03-probes/22-tune-moe.sbatch`](03-probes/22-tune-moe.sbatch) |
| prefill memory probe | [`03-probes/23-probe-prefill-mem.sbatch`](03-probes/23-probe-prefill-mem.sbatch) |
| Nano5 FP8 KV | [`04-nano5/nano5-kv-fp8.sh`](04-nano5/nano5-kv-fp8.sh) |
| Nano5 16,384 prefill | [`04-nano5/nano5-bigger-prefill.sh`](04-nano5/nano5-bigger-prefill.sh) |
| Nano5 ITL curve | [`04-nano5/nano5-itl-curve.sh`](04-nano5/nano5-itl-curve.sh) |
| 最終 NSCC submission | [`05-submissions/05-nscc-h100-sglang-2844/qwen3-pd-h100-16gpu.pbs`](05-submissions/05-nscc-h100-sglang-2844/qwen3-pd-h100-16gpu.pbs) |

## 實驗 1：PD 是否能啟動

| 設定 | 值 |
|---|---|
| Job／平台 | 214749／NSCC H100，8 GPU |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign driver） |
| Engine／模式 | PD；初始 baseline |
| P:D／GPU topology | —；8 GPU |
| 併發 N | 低併發；數值未記錄 |
| 主要參數 | 初始設定，尚未建立 P:D 基準 |
| 對照 | 無，第一次成功執行 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 410 | — | — | 可啟動並完成請求 |
| pass | 1.000 | — | — | 低併發，壓力不足 |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | — | — | — | — |

**影響／解讀**：證明 PBS、router 與 PD 路徑可用，但尚不足以判斷效能。下一步是測 P:D 比例。

## 實驗 2：4P + 12D

| 設定 | 值 |
|---|---|
| Job／平台 | 214750／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign driver） |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 4P + 12D |
| 併發 N | 24、32 |
| 主要參數 | 偏重 decode 的 PD topology |
| 對照 | 無；第一個形狀 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 986 @ N=24 | — | — | — |
| pass | 0.876 @ N=32 | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | 3,606 ms @ N=32 | — | — | 先超過門檻 |
| ITL p90 | — | — | — | — |

**影響／解讀**：4 張 prefill GPU 餵不飽 12 張 decode GPU；併發提高後先壞的是 TTFT。下一步改測 8P + 8D。

## 實驗 3：8P + 8D

| 設定 | 值 |
|---|---|
| Job／平台 | 214760／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign driver） |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 8P + 8D |
| 併發 N | 48 |
| 主要參數 | 改 topology，且併發也由 24/32 改為 48 |
| 對照 | 214750；非嚴格單變因 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | — | 986 | — | — |
| pass | 0.234 | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | 2,933 ms | 3,606 ms | −673 ms | 仍失守 |
| ITL p90 | 21.5 ms | — | — | 同時失守 |

**影響／解讀**：TTFT 與 ITL 同時失敗，問題不只是 prefill GPU 數量，下一步檢查 engine 的 parallelism 與通訊。

## 實驗 4：關閉 expert parallelism

| 設定 | 值 |
|---|---|
| Job／平台 | 214830／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant） |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 8P + 8D；TP4 worker |
| 併發 N | — |
| 主要參數 | 關閉 EP；不使用預設 `ep-size = TP size` |
| 對照 | 214760／原 EP 設定 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,240 | 約 986 | 約 2.27× | 最大單一增益 |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | — | — | — | EP all-to-all 成本較高 |

**影響／解讀**：16 GPU 規模下 EP 的 all-to-all 比 TP-style reduction 昂貴；這是「關掉預設」而非增加功能帶來的增益。

## 實驗 5：整數 GPU id

| 設定 | 值 |
|---|---|
| Job／平台 | 215267／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant） |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 8P + 8D |
| 併發 N | 56；並解鎖 N=64、72 |
| 主要參數 | `CUDA_VISIBLE_DEVICES` 由 UUID 改為整數 id |
| 對照 | 214830／UUID 設定 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,380 @ N=56；2,444 @ N=72 | 2,240（N=56 對照） | +6.3% @ N=56 | 不同 N 的分數不可直接互比 |
| pass | — | — | — | — |
| R (req/s) | 11.90 @ N=56 | 11.20 @ N=56 | +6.3% | 增益全部在 prefill |
| TTFT p90 | — | — | — | — |
| ITL p90 | — | — | — | — |

**影響／解讀**：PBS 的 UUID 被 `int()` 解析失敗後吞掉例外，collective fallback 到 NCCL；這是環境／launcher bug。

## 實驗 6：預熱 DeepGEMM cache

| 設定 | 值 |
|---|---|
| Job／平台 | 215459／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant）；相關預編譯工具為 [`03-probes/21-compile-deepgemm.sbatch`](03-probes/21-compile-deepgemm.sbatch) |
| Engine／模式 | SGLang PD；prefill DeepGEMM |
| P:D／GPU topology | 8P + 8D |
| 併發 N | — |
| 主要參數 | 預先暖 DeepGEMM cache |
| 對照 | 有／無 cache warm-up 的啟動行為 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 不可讀 | — | — | CUDA graph capture 仍失敗 |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | — | — | — | — |

**影響／解讀**：509 SU 的結果否定「只要預編譯就能修好」；觸發點是第一次 module load，下一步是在 prefill role 關閉 CUDA graph。

## 實驗 7：prefill 專用 DeepGEMM

| 設定 | 值 |
|---|---|
| Job／平台 | 215675／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant）；提交版參考 [`05-submissions/01-nscc-h100-sglang-2706/qwen3-pd-h100-16gpu.pbs`](05-submissions/01-nscc-h100-sglang-2706/qwen3-pd-h100-16gpu.pbs) |
| Engine／模式 | SGLang PD；DeepGEMM 僅 prefill |
| P:D／GPU topology | 2× TP4 prefill + 2× TP4 decode |
| 併發 N | 72；膝點由 64 移至 72 |
| 主要參數 | `MOE_RUNNER_BACKEND_PREFILL=deep_gemm`、`DISABLE_PREFILL_GRAPH=1` |
| 對照 | 215267／Triton prefill、同 N |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,706 | 2,444 | +10.7% | — |
| pass | 0.976 | — | — | 官方視窗 |
| R (req/s) | 13.53 | — | — | — |
| TTFT p90 | 1,371 ms | 2,669 ms | −49% | N=72 |
| ITL p90 | 19.23 ms | — | — | decode 不使用 DeepGEMM |

**影響／解讀**：大 batch prefill 適合 DeepGEMM，小 batch decode 反而慢 1.3–1.5×；PD 分離使兩個 role 能用不同 kernel。

**215675 的完整 sweep（原始 submission README 可追溯的欄位）**：

| N | requests | good output tokens/s | pass | R (req/s) | TTFT p90 | ITL p90 | 失敗觀察 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 56 | 1,512 | — | 0.983 |  — | — | —（ITL p99 19.09） | TTFT-only 26，前 10 s |
| 64 | 1,621 | 2,562 | 0.977 | 13.11 | 1,110 ms | —（ITL p99 19.53） | TTFT-only 34；ITL-only 3 |
| 72 | 1,721 | **2,706** | 0.976 | 13.53 good-request rate | **1,371 ms** | **19.23 ms**（p99 19.33） | TTFT-only 42，前 10 s |
| 80 | 1,722 | 2,682 | 0.970 | 13.41 good-request rate | 2,021 ms | —（ITL p99 19.41） | TTFT-only 52，前 10 s |

這裡的 `R` 原始紀錄同時提供 offered rate 與 good-request rate；上表優先列出能直接對應 score 的 good-request rate，避免把兩者混用。

## 實驗 8：vLLM 對照

| 設定 | 值 |
|---|---|
| Job／平台 | 215852／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（vLLM campaign cell） |
| Engine／模式 | vLLM PD |
| P:D／GPU topology | — |
| 併發 N | 48 |
| 主要參數 | 更換 engine；其他 engine 預設也隨之改變 |
| 對照 | SGLang；非單變因 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,196 | SGLang 2,706 | −510 | 不可歸因於單一 flag |
| pass | — | — | — | ITL gate 失敗 |
| R (req/s) | — | — | — | — |
| TTFT p90 | 486 ms | 1,872 ms | 較低 | — |
| ITL p90 | — | — | — | vLLM 的主要限制 |

**影響／解讀**：vLLM 掛在 ITL，SGLang 掛在相對較高的 TTFT；下一步把 GPU 從 prefill 移給 decode。

## 實驗 9：vLLM 1P + 3D

| 設定 | 值 |
|---|---|
| Job／平台 | 215864／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（vLLM campaign variant） |
| Engine／模式 | vLLM PD |
| P:D／GPU topology | 1P + 3D |
| 併發 N | — |
| 主要參數 | 只改 GPU 分配 |
| 對照 | 215852 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 分數下降 85× | 2,196 | −85× | — |
| pass | — | — | — | — |
| R (req/s) | 6.25 prefill capacity | — | 不隨 N 成長 | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | 15.1 ms | 較高 | 改善 | 單請求延遲不是 capacity |

**影響／解讀**：ITL 確實改善，但一個 TP4 prefill worker 餵不出足夠請求；證明 latency 不等於 capacity。

## 實驗 10：移植 H200 MoE config

| 設定 | 值 |
|---|---|
| Job／平台 | 215916／NSCC |
| 腳本 | [`03-probes/22-tune-moe.sbatch`](03-probes/22-tune-moe.sbatch) |
| Engine／模式 | SGLang PD；MoE Triton config |
| P:D／GPU topology | — |
| 併發 N | — |
| 主要參數 | 只更換 H200 調出的 MoE config |
| 對照 | H100 原生 config |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | — | — | — | 分數因 pass 崩潰 |
| pass | 0.090 | 0.990 | −0.900 | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | +5.3% | — | 變差 | — |

**影響／解讀**：Triton config 綁定 GPU 型號與 batch shape，H200 設定不能直接移植到 H100。

## 實驗 11：prefill batch 12,288（第一次）

| 設定 | 值 |
|---|---|
| Job／平台 | 215921／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant；記憶體分析見 [`03-probes/23-probe-prefill-mem.sbatch`](03-probes/23-probe-prefill-mem.sbatch)） |
| Engine／模式 | SGLang PD；prefill batch probe |
| P:D／GPU topology | — |
| 併發 N | — |
| 主要參數 | `CHUNKED_PREFILL=12288`、`MAX_PREFILL_TOKENS=12288`、`MEM_FRACTION=0.85` |
| 對照 | 控制組被變體解析 bug 破壞 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 不可讀 | — | — | OOM |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | — | — | — | — |

**影響／解讀**：`VARIANTS="dgpf+bigpf:..."` 中的 `+` 產生空變體，額外消耗 1,470 SU；OOM 只表示當時 memory allocation 不足。

## 實驗 12：all-reduce 實作

| 設定 | 值 |
|---|---|
| Job／平台 | 215960／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（paired benchmark driver）；記憶體觀察另見 [`03-probes/23-probe-prefill-mem.sbatch`](03-probes/23-probe-prefill-mem.sbatch)，分析也見 [`05-submissions/04-nscc-h100-sglang-2814/qwen3-pd-h100-16gpu.pbs`](05-submissions/04-nscc-h100-sglang-2814/qwen3-pd-h100-16gpu.pbs) |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 2× TP4 prefill + 2× TP4 decode |
| 併發 N | 56、64、72、80 |
| 主要參數 | 只改 all-reduce implementation |
| 對照 | 同一 allocation 的 paired control |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,712 | 2,710 | +0.07% | 否證有效增益 |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | 失敗集中開場 |
| ITL p90 | 單請求未越線 | — | — | — |

**影響／解讀**：all-reduce 沒有帶來有效提升；失敗約為 `N − 30`，問題轉向 prefill rate 與記憶體。

## 實驗 13：per-role memory fraction

| 設定 | 值 |
|---|---|
| Job／平台 | 216132／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant）；提交腳本見 [`05-submissions/04-nscc-h100-sglang-2814/qwen3-pd-h100-16gpu.pbs`](05-submissions/04-nscc-h100-sglang-2814/qwen3-pd-h100-16gpu.pbs) |
| Engine／模式 | SGLang PD；DeepGEMM prefill |
| P:D／GPU topology | 2× TP4 prefill + 2× TP4 decode |
| 併發 N | 72、80、88 |
| 主要參數 | prefill `MEM_FRACTION_PREFILL=0.80`；decode `0.85`；batch `12,288` |
| 對照 | paired control；原 prefill memory fraction |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,834 @ N=88 | — | +4.6% | N=88 後來不穩，不可直接宣告 |
| pass | 0.962 @ N=88 | — | — | 後續重跑為 0.880 |
| R (req/s) | 14.72 | 13.89 | +6.0% | prefill service rate |
| TTFT p90 | — | — | — | — |
| ITL p90 | — | — | — | — |

**影響／解讀**：prefill KV pool usage 僅 0.11–0.13、峰值 0.19；將 0.85 降至 0.80 讓 DeepGEMM workspace 保留 12,288 batch。

## 實驗 14：H100 原生 MoE config

| 設定 | 值 |
|---|---|
| Job／平台 | 216159／NSCC |
| 腳本 | [`03-probes/22-tune-moe.sbatch`](03-probes/22-tune-moe.sbatch)；tuner 原始程式為 [`03-probes/tuning_fused_moe_triton-v0.5.20.py`](03-probes/tuning_fused_moe_triton-v0.5.20.py) |
| Engine／模式 | SGLang PD；H100-native MoE config |
| P:D／GPU topology | 2× TP4 prefill + 2× TP4 decode |
| 併發 N | 88 |
| 主要參數 | 只換自行調出的 H100／目標 batch config |
| 對照 | 同兩節點控制組；216132 的 N=88 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 712 | 2,742 | −74.0% | 不能提交 |
| pass | — | 0.880 | — | 1,317/1,790 失敗 |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | 20.32 ms | 19.03 ms | +6.8% | — |

**影響／解讀**：config 反向傷害；控制組本身也由 216132 的 0.962 變成 0.880，說明 N=88 的 run-to-run variance 很大。

**216132／216159 的完整 N 對照**：

| N | 216132 good tokens/s | 216132 pass | 216159 good tokens/s | 216159 pass | ITL 觀察 |
|---:|---:|---:|---:|---:|---|
| 72 | 2,686 | — | — | — | — |
| 80 | **2,814** | 0.960 | **2,742** | 0.930 | — |
| 88 | **2,834** | 0.962 | **2,608** | **0.880** | 216132 max 19.91 ms；216159 有 153 筆 ITL failure |

216132 的 paired 控制組在 N=88 曾記錄 `0.962`，但 216159 對相同設定重跑得到 `0.880`；因此 N=88 的 2,834 不能作為穩定或可提交數字。

## 實驗 15：TTFT 組成分析

| 設定 | 值 |
|---|---|
| Job／平台 | 216179／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch) 產生 metrics；分析既有 histogram |
| Engine／模式 | SGLang PD；非新 benchmark |
| P:D／GPU topology | 2× TP4 + 2× TP4 |
| 併發 N | — |
| 主要參數 | `--enable-metrics`；不改 deployment |
| 對照 | 無；observability analysis |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | — | — | — | 非 benchmark |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | 1,231 ms | — | — | bootstrap 55 ms、forward 785 ms、KV transfer 391 ms |
| ITL p90 | — | — | — | — |

**影響／解讀**：KV transfer 約佔 TTFT 32%，因此下一步測連線暖機與縮小 KV 傳輸，而不是只調 model forward。

## 實驗 16：兩階段連線暖機

| 設定 | 值 |
|---|---|
| Job／平台 | 216202／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)；最終實作參考 [`05-submissions/05-nscc-h100-sglang-2844/qwen3-pd-h100-16gpu.pbs`](05-submissions/05-nscc-h100-sglang-2844/qwen3-pd-h100-16gpu.pbs) |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 2× TP4 prefill + 2× TP4 decode |
| 併發 N | 80、88 |
| 主要參數 | burst 32 @ C=32；sequential 8 @ C=1；ISL 4,000、OSL 16 |
| 對照 | warm-up 較少的既有執行 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,844 @ N=80 | — | — | 當時最佳 |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | N=88 失敗 0 筆 | 69 筆 | −69 | 尚未證明因果 |

**影響／解讀**：看似改善 ITL tail，但必須由 paired rerun 排除自然波動。

## 實驗 17：暖機 paired rerun

| 設定 | 值 |
|---|---|
| Job／平台 | 216211／NSCC |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant） |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 2× TP4 + 2× TP4 |
| 併發 N | 88 |
| 主要參數 | 沿用兩階段 warm-up；控制組同樣 warm-up |
| 對照 | 有；同條件 paired control |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | — | — | — | — |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | 控制組 95 筆失敗 | 216202 為 0 筆 | 不可歸因 | 收回 warm-up 修復 ITL 的結論 |

**影響／解讀**：暖機可保留作為 cold-start protection，但不能宣稱它提升 score 或修復 ITL tail。

## 實驗 18：`--enable-torch-symm-mem`

| 設定 | 值 |
|---|---|
| Job／平台 | 216243／NSCC（216204 是未配對早期觀察） |
| 腳本 | [`02-campaign/10-campaign.sbatch`](02-campaign/10-campaign.sbatch)（campaign variant） |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 2× TP4 + 2× TP4 |
| 併發 N | 80 |
| 主要參數 | 只開 `--enable-torch-symm-mem` |
| 對照 | 同兩節點、同一 job，控制組先跑 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 2,748 | 2,844 | −3.4% | paired |
| pass | — | — | — | — |
| R (req/s) | 14.19 | 14.64 | −3.1% | throughput 下降 |
| TTFT p90 | — | — | — | — |
| ITL p90 | 18.61 ms；max 19.42 ms | 19.22 ms；max 20.10 ms | 改善 | tail 變短 |

**影響／解讀**：ITL 變好但 throughput 變差；整體是「拿吞吐換未失敗門檻餘裕」的淨負面。

**同 allocation 的額外結果**：N=72 的 `bigpf` 有 30 筆 ITL failures、單請求 max
20.54 ms；N=80 有 20 筆、max 20.32 ms；控制組在相同 N 沒有 ITL failures、max 不超過 19.47 ms。

## 實驗 19：ITL(batch) 曲線

| 設定 | 值 |
|---|---|
| Job／平台 | 373046／Nano5，單節點 8 GPU |
| 腳本 | [`04-nano5/nano5-itl-curve.sh`](04-nano5/nano5-itl-curve.sh) |
| Engine／模式 | SGLang 真 PD，`4x1:4x1` |
| P:D／GPU topology | 1× TP4 prefill + 1× TP4 decode |
| 併發 N | 4、8、12、16、20、24、28、32、40、48、64 |
| 主要參數 | prefill DeepGEMM；batch 12,288；prefill memory fraction 0.80；短 20/20/20 s window |
| 對照 | 同一設定掃不同 batch；非 A/B |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | — | — | — | probe，不是正式分數 |
| pass | — | — | — | — |
| R (req/s) | — | — | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | `6.79 + 3.577 × ln(batch)` | — | — | `R²=0.989`；餘裕 11–37% |

**影響／解讀**：ITL 曲線是凹的，batch 增加到 expert coverage 飽和後斜率下降；下一步測 FP8 KV 與 16,384 prefill。

## 實驗 20：FP8 KV cache

| 設定 | 值 |
|---|---|
| Job／平台 | 372421／Nano5 |
| 腳本 | [`04-nano5/nano5-kv-fp8.sh`](04-nano5/nano5-kv-fp8.sh) |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 2× TP4 prefill + 2× TP4 decode；2 nodes |
| 併發 N | 80、88 |
| 主要參數 | `KV_CACHE_DTYPE=fp8_e4m3`；batch 12,288；prefill fraction 0.80 |
| 對照 | BF16 KV paired control；第二格被 port 9100 吃掉 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 不可讀 | 不可讀 | — | port collision |
| pass | 不可讀 | 不可讀 | — | — |
| R (req/s) | 不可讀 | 不可讀 | — | — |
| TTFT p90 | — | — | — | transfer 408 → 186 ms |
| ITL p90 | 16.77 ms | 18.48 ms | −1.71 ms | 機制已證實 |

**影響／解讀**：KV pool 140,076 → 280,153、transfer 408 → 186 ms，但 score 不可讀；修 port 後才可宣稱整體增益。

## 實驗 21：prefill batch 16,384

| 設定 | 值 |
|---|---|
| Job／平台 | 372456／Nano5 |
| 腳本 | [`04-nano5/nano5-bigger-prefill.sh`](04-nano5/nano5-bigger-prefill.sh) |
| Engine／模式 | SGLang PD |
| P:D／GPU topology | 2× TP4 prefill + 2× TP4 decode；2 nodes |
| 併發 N | 80、88 |
| 主要參數 | `CHUNKED_PREFILL=16384`、`MAX_PREFILL_TOKENS=16384`、`MEM_FRACTION_PREFILL=0.74` |
| 對照 | 12,288／0.80 paired control；第二格被 port 9100 吃掉 |

| 結果指標 | 實驗值 | 控制／基準 | 差異 | 備註 |
|---|---:|---:|---:|---|
| good output tokens/s | 不可讀 | 不可讀 | — | port collision |
| pass | 不可讀 | 不可讀 | — | — |
| R (req/s) | 不可讀 | 不可讀 | — | — |
| TTFT p90 | — | — | — | — |
| ITL p90 | — | — | — | — |

**影響／解讀**：log 的 `reducing max_m to 16384` 證實 memory arithmetic，但沒有證明 throughput 或 score 提升。根因是 cell 1 使用 port 9100，而該 port 是 Nano5 的 Prometheus IPMI exporter；修正 `pd_assert_ports_free()` 後重跑。

## 總結：目前可以用來比較的結果

| 實驗 | good tokens/s | pass | R | TTFT p90 | ITL p90 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 214749 PD baseline | 410 | 1.000 | — | — | — | 可啟動，不是壓力基準 |
| 214750 4P+12D | 986 @ N=24 | 0.876 @ N=32 | — | 3,606 ms | — | prefill 不足 |
| 214830 no EP | 2,240 | — | — | — | — | 有效增益，約 2.27× |
| 215267 integer GPU id | 2,444 | — | 11.90 | — | — | 有效增益，+6.3% |
| 215675 prefill DeepGEMM | 2,706 | 0.976 | 13.53 | 1,371 ms | 19.23 ms | 有效增益，+10.7% |
| 215852 vLLM | 2,196 | — | — | 486 ms | — | TTFT 好但 ITL-bound |
| 216132 memory fraction | 2,834 @ N=88 | 0.962 | 14.72 | — | — | N=88 不穩，不可直接宣告 |
| 216243 symm mem | 2,748 | — | 14.19 | — | 18.61 ms | paired score −3.4% |
| 372421 FP8 KV | 不可讀 | 不可讀 | 不可讀 | — | 16.77 ms | 修 port 後重跑 |
| 372456 prefill 16K | 不可讀 | 不可讀 | 不可讀 | — | — | 修 port 後重跑 |

目前正式可提交的完整 deployment 參數，應以 [`05-submissions/05-nscc-h100-sglang-2844/qwen3-pd-h100-16gpu.pbs`](05-submissions/05-nscc-h100-sglang-2844/qwen3-pd-h100-16gpu.pbs) 為準；該提交在 N=80 測得 **2,844 good output tokens/s、pass=0.971、R=14.64 req/s、TTFT p90=1,899 ms、ITL p90=19.22 ms**。

## 數值完整性說明

目前 repository 沒有把每一個 job 的原始 `profile_export_aiperf.csv` 或完整 log 一起放進來；可補出的數字來自腳本註解、`.md` 摘要與 submission README。因此仍有些欄位只能標為 `—`，不能安全地從 score 反推 TTFT 或 ITL。若要把 21 個實驗全部補成每個 N 的完整結果，還需要把各 job 的 `runs/<jobid>/` 或 AIPerf CSV 匯入專案。
