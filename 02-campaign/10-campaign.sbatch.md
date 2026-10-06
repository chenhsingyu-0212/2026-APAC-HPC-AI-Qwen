# `10-campaign.sbatch` — 主力：一個 allocation 裡掃完所有設定

**跑過：** 大部分的 job。NSCC 的 214749–216243、Nano5 的 372316–373046。

一個 allocation 裡依序跑多個 **cell**（engine × shape），每個 cell 帶起一次部署、
掃完所有併發、拆掉。`VARIANTS` 讓同一個 shape 跑多個設定組合。

**為什麼是這個形狀：** NSCC/Nano5 拿到 allocation 是難的部分，而 checkpoint 載入要 12 分鐘 ——
所以「抓住 allocation 在裡面掃完」。Firmus 相反（17 分鐘硬上限），所以那邊用
`sweep-pd.sbatch`，一個 job 一個拓撲。

**`VARIANTS` 的語法有兩個地雷，各花過一次錢：**
- `+` 分隔**變體**，`,` 分隔**賦值**。`"dgpf+bigpf:..."` 會跑一個空的 `dgpf`（約 1,470 SU）
- 然後它 **word-split**，所以值裡有空白會被拆開（2,576 SU）。現在有守衛，而**第一版守衛
  跑在 word-split 之後，永遠不可能觸發**

**兩個現在才有的守衛：**
- `pd_assert_ports_free()` —— 載入 checkpoint 前實測綁定每個 port。port 9100 是 Nano5 的
  Prometheus IPMI exporter，吃掉了兩個實驗
- 暖機的錯誤**比例**（預設 10%）—— 原本只在「完全沒有成功回應」時中止，所以一個暖機
  32 個錯 14 個的部署照樣被量了約 90 SU
