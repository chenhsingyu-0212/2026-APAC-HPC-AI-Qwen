# `03-smoke-serve-aiperf.sbatch` — 客戶端能不能跟引擎對話

**跑過：** 每個叢集一次。

`02-smoke-serve` 加上一次短的 AIPerf。確認 tokenizer、endpoint 形狀、SSE 串流都對。

**它抓到的：** 這個 AIPerf 沒有 `--cache-bust`，所以前綴重用必須從**引擎端**關掉
（`--disable-radix-cache` / `--no-enable-prefix-caching`）。那是從另一側達成同一個狀態，
而且在 server 自己的 log 裡看得到 —— 客戶端旗標從來不會。
