# `05-pd-smoke.sbatch` — PD 的兩半能不能配對

**跑過：** 每個叢集一次，在第一個 campaign 之前。

一個 prefill worker、一個 decode worker、一個 gateway，送幾個請求。
**這是最便宜的「KV 傳輸路徑通不通」測試。**

**它抓到的：** 冷的 PD 佈局上，一對 prefill–decode 的第一個請求可以卡好幾分鐘 ——
那就是後來 `10-campaign.sbatch` 裡兩階段連線暖機存在的原因。
