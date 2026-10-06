# `nano5-build-image.sbatch` — 在 compute node 上建 SGLang 映像

**跑過：** job 372144（取消）、**372289（成功，13 分 12 秒，17.4 GB）**。

login node 建不了 —— enroot import 死在 `failed to set capabilities: Operation not permitted`。
八月隊友的 TensorRT-LLM 映像也是在 job 裡建的，同一個原因。

**372144 為什麼失敗（這條值錢）：** `SINGULARITY_TMPDIR` 放在 `/work`（NFS），
48 分鐘後寫入速度 **1 MB/s、進程 CPU 3%**。而 `/work` 不慢 —— 同節點 dd 實測 1.7 GB/s，
`/tmp` 2.4 GB/s。**解 container rootfs 是幾十萬個小檔加每檔一次 `setxattr`，
在 NFS 上每個都是一次 round trip。** 改放本機 `/tmp`：**118 MB/s、CPU 105%**。

`CACHEDIR` 刻意留在 `/work`：18 GB 的 layer blob 已經抓好，重跑會跳過下載。

`v0.5.20-cu129` 在 Docker Hub 不存在（MANIFEST_UNKNOWN），`v0.5.19-cu129` 存在。
