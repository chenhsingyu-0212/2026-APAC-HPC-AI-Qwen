# `make-down-config.py` — 產生 down projection 的 config，不需要 tuner

**從未送出。** 存在的理由值得記：

SGLang 的 `_sep` tuner 五次都沒產出。**但那支 tuner 不是必要的** ——
down GEMM 今天已經在跑 up 的切塊，所以一份數值完全相同、只多 `USE_TMA: true` 的
`_down.json` 就是一個**單變數實驗**。

**373046 之後降級**：它屬於「拿吞吐換 ITL 餘裕」那一類，而 symm 已經示範那在飽和點上是淨虧。

**11 個測試。**
