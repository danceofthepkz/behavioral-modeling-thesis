# 在 Firebird 上运行 rl_ladder

每一节注明在哪里执行。用户名沿用 `kpeng1-swat`，partition 沿用 `unowned`（和 rl_firebird_v2 一样）。

## 0. 前提

- `~/rl_firebird_v2/data/` 已经在 Firebird 上（v2 的包里带着）。`activate.sh` 会自动找到它，不需要重新上传数据。
- `~/rl_firebird_v2/validation/reference_legacy.json` 用于复现检查，也会被自动找到。
- 如果数据放在别处：`export RL_LADDER_DATA=/path/to/data`。

## 1. Mac：打包并上传代码

```bash
cd /Users/danceofthepkz/Desktop/00_Inbox/cogsthesis/behavioral-modeling-thesis/rl_ladder
scripts/pack_for_firebird.sh
scp dist/rl_ladder_*.tar.gz kpeng1-swat@firebird.swarthmore.edu:~/
```

包里只有代码和配置，没有数据和结果。

## 2. Firebird 登录节点：解压、准备环境

```bash
tar -xzf ~/rl_ladder_*.tar.gz -C ~/
cd ~/rl_ladder
```

环境二选一：

**A. 复用 `mice-rl`（推荐，最快）。** 它只缺 PyYAML 和 pytest：

```bash
scripts/setup_env.sh mice-rl
export RL_LADDER_ENV=$HOME/.conda/envs/mice-rl     # 建议写进 ~/.bashrc
```

**B. 新建独立环境 `rl-ladder`：**

```bash
scripts/setup_env.sh conda
```

然后检查：

```bash
source scripts/activate.sh            # 应打印 python 路径和 RL_LADDER_DATA=~/rl_firebird_v2/data
python -m rl_ladder list models
```

## 3. Firebird：在计算节点上跑测试

```bash
mkdir -p logs
sbatch --partition=unowned scripts/slurm/cmd.sbatch scripts/test.sh
```

用 `sacct -j <作业号> --format=JobID,State,ExitCode` 和 `logs/rl-ladder-cmd-<作业号>.out` 确认 **94 passed**。数据被找到时，两个复现测试也会运行，不会被 skip。

## 4. Firebird：试跑（pilot）

```bash
scripts/slurm/submit.sh smoke --chunk 5 --time 00:10:00
```

这会提交一个 array 任务和一个 collect 任务。完成后看 `logs/rl-smoke-collect-*.out`，应显示 `5/5 units complete`。

## 5. 正式运行（建议顺序）

实测耗时：19 只鼠 × 10 个模型 × 2 个窗口的全部拟合，一共约 250 CPU 秒。所以真实数据的拟合在 Mac 上跑就够了。Firebird 主要用在 model recovery 这类模拟加重拟合的大批量任务，以及加大 `n_reps` / `n_starts` 的时候。


| 顺序 | 命令 | 规模 | 建议 |
|---|---|---|---|
| 1 | `scripts/slurm/submit.sh replicate_paper --chunk 10 --time 00:30:00` | 38 units | 也可以直接在 Mac 本地跑（几秒钟） |
| 2 | `scripts/slurm/submit.sh core_ladder --chunk 10 --time 01:00:00` | 380 units | 实测：本地 12 核 25 秒（约 250 CPU 秒），本地跑就够 |
| 3 | `scripts/slurm/submit.sh param_recovery --chunk 20 --time 01:00:00` | 1000 units | |
| 4 | `scripts/slurm/submit.sh model_recovery --chunk 2 --time 02:00:00` | 500 units，每个 unit 拟合 10 个模型 | **必须在 core_ladder 完成之后**：它用 `param_source: fitted`，会读取 core_ladder 缓存的拟合 |
| 5 | `scripts/slurm/submit.sh sensitivity --chunk 10 --time 01:00:00` | 1140 units | |
| 6 | `scripts/slurm/submit.sh ppc --chunk 5 --time 01:00:00` | 114 units | 复用 core_ladder 的拟合 |

任何设置都可以在命令末尾覆盖，例如 `--validate.n_reps=20` 或 `--fit.n_starts=40`。如果要和默认结果分开存，加 `--name-suffix _40starts`。

先只交一个 array index 看看耗时：

```bash
scripts/slurm/submit.sh model_recovery --chunk 2 --pilot
```

## 6. 监控、续跑、汇总

```bash
squeue -u $USER
sacct -j <array作业号> -X --format=JobID,State,Elapsed,ExitCode | sort -k2 | uniq -c -f1
```

- **有 unit 失败或超时**：再执行一次同样的 `submit.sh` 命令即可，已完成的 unit 会自动跳过。失败原因（traceback）在 `results/runs/<run>/failures/`。
- **手动汇总**：`python -m rl_ladder collect plans/<plan>.json`。如果还有缺失，它会拒绝写汇总；要先看部分结果，加 `--allow-partial`，文件名会带 `partial_` 前缀。

## 7. Mac：下载结果

```bash
scp -r kpeng1-swat@firebird.swarthmore.edu:~/rl_ladder/results/runs/core_ladder \
    /Users/danceofthepkz/Desktop/00_Inbox/cogsthesis/behavioral-modeling-thesis/rl_ladder/results/runs/
```

`summary/` 里是汇总表。`results/` 在 `.gitignore` 里，不会被提交到公开 repo。

## 注意事项

1. **登录节点只做轻活。** `plan`、`count`、`collect`、`list` 可以在登录节点跑；拟合一律走 `sbatch`。
2. **跑的过程中不要改代码。** 缓存 key 里包含科学代码（`models/`、`fit/`、`data.py` 等）的校验和。代码一改，所有 key 都会变，结果会重新计算。这是有意设计的，但如果在作业中途改代码，同一个 run 里就会混入两个版本。
3. **重复计算无害。** 同时运行的 array 任务可能都需要同一个父模型（比如 S0），于是各算一遍。结果是确定性的，写入是原子操作，只会浪费一点 CPU。
4. **`unowned` 可能被抢占。** 被抢占的 unit 重新提交即可续跑。
5. **array 上限。** 用 `scontrol show config | grep MaxArraySize` 查看。超了就把 `--chunk` 调大。
6. **numba 缓存。** 每个作业用自己的临时目录（`unit.sbatch` 里设置了 `NUMBA_CACHE_DIR`），不会在共享 home 上互相写坏。每个 array 任务开头会有几秒编译时间，这是正常的。
7. **结果存放位置。** 默认在 `~/rl_ladder/results`。如果 home 配额紧，可以 `export RL_LADDER_RESULTS=/scratch/...`（`activate.sh` 不会覆盖已设置的值）。
8. **数据隐私。** `new5`（AS 小鼠）是未发表数据。不要把它的结果放进公开 repo 或网站。
9. **模型族的提醒。** S 族的 `beta_q` 在小鼠数据上不可识别（只有 `alpha*beta_q` 能被约束住）。看 S 族的参数之前，先看 `fits.csv` 的 `at_bound` 列和 param_recovery 的结果。另有一组 `S_fixed` 模型可选。
