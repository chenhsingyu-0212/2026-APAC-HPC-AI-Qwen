# `nano5-smoke.sh` — Nano5 的第一次 16 卡嘗試

**送出為 372316，使用者要求取消改送 A/B。**

跑 2,844 的設定、短視窗 —— 測的是**平台**不是設定。

**它抓到的（在送出時就抓到）：** `dev` 的 `MaxNodes=1`。sbatch 回
`Node count specification invalid`，**不指名 partition 也不指名規則**。
而 `sbatch --test-only` **不檢查 MaxNodes** —— 它接受了 `--nodes=2` on dev。
`normal` 是 `MaxNodes=2`，正好是題目要的 16 卡，也是這台機器的上限。
