# `23-probe-prefill-mem.sbatch` — 量 prefill 側到底用了多少 KV pool

**跑過：** job 215960 的一部分。

**它產出了 +6.0% 的那個數字。** 在約 1,250 個 prefill step 上，pool 的 `token usage`
坐在 **0.11–0.13、峰值 0.19** —— 包含每一筆 TTFT 失敗所在的開場十秒。

**pool 比這個 job 需要的大五倍，而它餓死的正是我們要的東西：**
DeepGEMM 要 `max_m = 2 × max_prefill_tokens` 然後對半砍到放得下。
`mem_fraction` 0.85 時 12,288 的批次砍兩次變 6,144；**0.80 時只砍一次，剛好 12,288**。
