# `22-tune-moe.sbatch` — 調 Triton 的 MoE config

**跑過：** job 215916（移植 H200 的）、216159（自己調的）、99894 等五次 sep tuner。

**兩次都反向傷害：** 215916 pass 0.990 → 0.090；216159 ITL 19.03 → 20.32、分數 712 對 2,742。

**三個地雷，各踩過一次：**
- `triton_version` 這個 key 會讓 SGLang 的 loader 壞掉 → `install-moe-config.sh` 會剝掉它
- `SGLANG_MOE_CONFIG_DIR` **取代整棵樹**而不是疊加 → 要用 symlink 鏡像再覆蓋
- 檔名裡 `block_shape=[128, 128]` **中括號內的空格是必須的** —— 我自己在 scp 時改名踩過

**sep tuner 五次都沒產出**，最後一次搜完 1,920 個組態、exit 0、零檔案，根因不明。已擱置。
