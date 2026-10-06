# `tuning_fused_moe_triton-v0.5.20.py` — SGLang 自帶的 MoE tuner（0.5.20 版）

**原封不動從 SGLang 複製過來的**，版本號寫在檔名裡，因為它的輸出格式會隨版本變
而 config 的檔名又綁 Triton 版本目錄。

job 216159 用它調出來的 config **讓分數從 2,742 掉到 712**（ITL 19.03 → 20.32，
1,317/1,790 筆失敗）。
