#!/usr/bin/env bash
# Submit an experiment as a Slurm job array, plus a collect job that runs after it.
#   scripts/slurm/submit.sh EXPERIMENT [options] [--stage.key=value ...]
# Options
#   --partition P     Slurm partition (default: $RL_LADDER_PARTITION, else "unowned")
#   --time T          wall time per array task (default 01:00:00)
#   --mem M           memory per array task (default 2G)
#   --chunk K         units per array task (default 10; use 1-2 for model_recovery)
#   --max-running M   at most M array tasks at once (default 20)
#   --name-suffix S   appended to the run name (a separate results folder)
#   --pilot           submit array index 0 only, without the collect job
#   --dry-run         write the plan and print the sbatch commands, submit nothing
# Run from anywhere; jobs run from the project folder. Resubmitting the same
# experiment skips units whose results already exist.
set -eo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ $# -lt 1 ]]; then sed -n '2,15p' "$0"; exit 1; fi
exp="$1"; shift
partition="${RL_LADDER_PARTITION:-unowned}"; time="01:00:00"; mem="2G"; chunk=10
maxrun=20; suffix=""; pilot=0; dry=0; overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) partition="$2"; shift 2 ;;
        --time) time="$2"; shift 2 ;;
        --mem) mem="$2"; shift 2 ;;
        --chunk) chunk="$2"; shift 2 ;;
        --max-running) maxrun="$2"; shift 2 ;;
        --name-suffix) suffix="$2"; shift 2 ;;
        --pilot) pilot=1; shift ;;
        --dry-run) dry=1; shift ;;
        --*.*) overrides+=("$1"); shift ;;
        *) echo "Unknown option $1"; exit 1 ;;
    esac
done
source "$HERE/scripts/activate.sh" >/dev/null
cd "$HERE"
mkdir -p logs plans
plan="plans/${exp}${suffix}_$(date +%Y%m%d_%H%M%S).json"
python -m rl_ladder plan "$exp" --name-suffix "$suffix" --out "$plan" "${overrides[@]}"
n=$(python -m rl_ladder count "$plan" --chunk-size "$chunk")
array="0-$((n - 1))%${maxrun}"
[[ $pilot -eq 1 ]] && array="0"
unit_cmd=(sbatch --parsable --partition "$partition" --time "$time" --mem "$mem"
          --array "$array" --job-name "rl-${exp}${suffix}"
          scripts/slurm/unit.sbatch "$plan" "$chunk")
collect_cmd() {  # $1 = dependency
    echo sbatch --parsable --partition "$partition" --time 00:30:00 --mem 4G \
        --dependency="afterany:$1" --job-name "rl-${exp}${suffix}-collect" \
        scripts/slurm/collect.sbatch "$plan"
}
echo "plan: $plan  ($n array tasks of up to $chunk units)"
if [[ $dry -eq 1 ]]; then
    echo "dry run:"; echo "  ${unit_cmd[*]}"
    [[ $pilot -eq 1 ]] || echo "  $(collect_cmd '<array job>')"
    exit 0
fi
jid=$("${unit_cmd[@]}")
echo "array job: $jid"
if [[ $pilot -eq 0 ]]; then
    cid=$(sbatch --parsable --partition "$partition" --time 00:30:00 --mem 4G \
          --dependency="afterany:$jid" --job-name "rl-${exp}${suffix}-collect" \
          scripts/slurm/collect.sbatch "$plan")
    echo "collect job: $cid (runs after the array finishes)"
fi
echo
echo "monitor:  squeue -j $jid"
echo "          sacct -j $jid -X --format=JobID,State,Elapsed,ExitCode | tail"
echo "logs:     logs/rl-${exp}${suffix}-${jid}_<index>.out"
echo "collect:  python -m rl_ladder collect $plan     (or see logs/*collect*)"
