# `probe-prefill-mem-inner.sh` — 記憶體探針在節點上跑的那半

`23-probe-prefill-mem.sbatch` 的內層。拆開是因為它要在 srun step 裡跑，
而外層負責排程與產出路徑。

讀的是引擎自己報的 `token usage` 與 `max_total_num_tokens`，**不是** `nvidia-smi` ——
後者看不到 pool 裡有多少被用掉。
