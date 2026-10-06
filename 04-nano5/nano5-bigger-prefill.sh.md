# `nano5-bigger-prefill.sh` — 16,384 的 prefill 批次，配對

**跑過：** job 372456（同一個 port bug → 不可讀）。

**記憶體算術驗證了**：log 有 `reducing max_m to 16384`，所以 `MEM_FRACTION_PREFILL=0.74` 正確。

**那個 0.74 是算出來的不是猜的。** 唯一的實測錨點是 f=0.80、批次 12,288 時
DeepGEMM 要 2×=24,576 然後**對半砍恰好一次**，所以
`required(12288) ≤ available(0.80) = 13.73 GB < required(24576)`。
把 required 當線性，這組不等式推出 16,384 需要的 f 落在 **0.743–0.857**，
0.74 在整個區間內都成立，0.75 是邊緣。

代價是 pool 剩約 3.7 GB，對實測峰值 0.6 GB —— 6 倍餘裕。
