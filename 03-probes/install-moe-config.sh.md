# `install-moe-config.sh` — 把一份 MoE config 裝到引擎看得到的地方

處理上面那三個地雷：剝掉 `triton_version`、用 symlink 鏡像整棵包裝好的樹再覆蓋、
把檔名正規化成帶空格的 `[128, 128]`。`MERGE_BASE` 可以疊在現有的上面。

**23 個測試**（`tests/test-install-moe-config.sh`）。
