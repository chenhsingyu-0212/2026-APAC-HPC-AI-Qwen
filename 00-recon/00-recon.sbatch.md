# `00-recon.sbatch` — 探查一個叢集能給什麼

**跑過：** NYCU、Firmus、NSCC 各一次（最早期）。

讀出 partition、GPU 型號與數量、QOS 的 GrpTRES/MaxTRES、容器 runtime 是否存在、
python 版本、CUDA 是否在 PATH 上。

**為什麼需要它：** NYCU 設了 `PrivateData=jobs,nodes,usage` 加 AdminLevel=None，
所以 `sinfo`、`scontrol show node`、別人的 job 全部看不到 —— **那台機器的所有事實
都是靠送 job 讀回來的**。`run/sites/*.sh` 裡每一行註解的來源就是這支。
