# `00-recon.sh` — recon 的實際內容

`00-recon.sbatch` 在節點上執行的部分。拆開是因為它也要能在 login node 上直接跑
（有些事實在 login node 就看得到，不必付一個 job 的錢）。
