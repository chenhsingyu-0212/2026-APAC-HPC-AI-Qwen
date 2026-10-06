# `submit-site.sh` — 把 sbatch/qsub 的差異關在一個地方

**每次送 job 都經過它。**

讀 `lib/sites/<site>.sh`，組出那個叢集接受的參數。四個叢集要四種不同的寫法：

| 叢集 | GPU 參數 | 其他強制 |
|---|---|---|
| NSCC | PBS `-l select=2:ncpus=112:ngpus=8` | `-P <project>` |
| NYCU | `--gres=gpu:gpu:8`（型別要寫兩次） | — |
| Firmus | `--gres=gpu:8` | — |
| **Nano5** | **`--gpus-per-node=8`**，`--gres` 被拒 | **`--account` 必填**、CPU ≤ 12/卡 |

送出前印**價格**（`<site>_su`）。Nano5 的錢包是共用的，16 卡一小時 192 SU。
