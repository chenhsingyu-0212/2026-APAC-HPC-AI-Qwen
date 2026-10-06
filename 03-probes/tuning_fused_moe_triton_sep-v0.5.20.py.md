# `tuning_fused_moe_triton_sep-v0.5.20.py` — 分開調 up/down GEMM 的那一版

`_sep` 版本，分開調 up 和 down projection。**五次都沒產出，已擱置。**

`make-down-config.py` 繞過了它：down GEMM 本來就在用 up 的切塊，所以一份只多
`USE_TMA: true` 的檔就是單變數實驗。
