# `nano5-itl-curve.sh` — ITL 對 decode batch 的曲線

**跑過：** 372908、372918、372921、372975 各失敗一次（**全部 0 SU**），
**373046 成功（19 分 26 秒、31 SU、11 個點）**。

**它回答的問題決定了後面每一項值不值得排。** 結果：
`ITL = 6.79 + 3.577 × ln(batch)`，R² 0.989，**凹的**。decode 餘裕 **11–37%**，
不是我先前發表的 98%。而這支 8 卡的探針預測得了 16 卡部署在操作點的 ITL，誤差 ±0.5 ms。

**兩個設計決定是重點：**
- **單節點真 PD（`4x1:4x1`）不是 solo。** solo 模式下同一個引擎既 prefill 又 decode，
  而 785 ms 的 prefill forward 在每個正在 decode 的請求的 token 流裡就是 785 ms 的間隙 ——
  AIPerf 量到的 ITL 會被 prefill 停頓主導
- **TP4 不是 TP8。** TP8 的每卡權重砍半，截距會跟著砍半，就量不到計分部署的那條曲線

**四次失敗各揭露一個不同的接線錯誤**，每一個都會在 `normal` 上吃掉六小時：
缺 AIPerf kit、`AIPERF` vs `AIPERF_BIN`、`.sif` 被餵給 pyxis、
`declare` 在被函式 source 的檔案裡變成區域變數。
