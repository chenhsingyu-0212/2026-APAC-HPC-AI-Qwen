# `01-smoke-image.sbatch` — 容器映像能不能用

**跑過：** NYCU、Firmus。

只做一件事：進容器、`import` 引擎、印版本。**不載模型。**

**它抓到的：** NYCU 完全沒有容器 runtime（singularity/apptainer/enroot/podman/docker 全缺），
所以三個引擎都改走 pip venv。這件事如果等到載完 221 GB 才發現，代價是一個 allocation。
