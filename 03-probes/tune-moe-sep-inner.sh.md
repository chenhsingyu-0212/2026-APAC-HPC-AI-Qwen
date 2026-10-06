# `tune-moe-sep-inner.sh` — sep tuner 在節點上跑的那半

**五次都沒產出。** 最後一次（job 99894）搜完 1,920 個組態、跑 45 分 32 秒、
**exit 0、寫出零個檔案**，根因不明。

**失敗的順序值得記：** `ModuleNotFoundError: ray` → `ModuleNotFoundError: common_utils`
→ **我自己寫的前置檢查在沒有 CUDA_HOME 的情況下失敗**（一個為了省節點而寫的守衛，
自己死在真正執行時會解決的 import 上）→ trialq 的 30 分鐘上限 → 然後是那個零檔案。

**已擱置** —— 不是因為這些失敗，而是因為 `make-down-config.py` 證明那支 tuner 不是必要的。
