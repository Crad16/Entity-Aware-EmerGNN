#!/usr/bin/env bash
# Launch one EmerGNN run in its own tmux session on an idle GPU.
#
#   scripts/launch.sh <run_name> <TWOSIDES|DrugBank> [GPU=auto|<idx>] -- <evaluate.py args...>
#
# Example:
#   scripts/launch.sh ts_S1_1_learned_s0 TWOSIDES auto -- --dataset=S1_1 --n_epoch=40 --epoch_per_test=2
#
# Logs go to runs/<run_name>.log; attach with `tmux attach -t <run_name>`.
# EXCLUDE_GPUS="3 4" keeps auto-pick off GPUs used by other work (e.g. someone else's jobs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=$1; DATASET_DIR=$2; GPU=${3:-auto}
shift 3; [ "${1:-}" = "--" ] && shift

if tmux has-session -t "$NAME" 2>/dev/null; then
    echo "tmux session '$NAME' already exists" >&2; exit 1
fi
# GPUs our live runs already occupy (from each run log's header line)
USED=$(for log in "$ROOT"/runs/*.log; do
    s=$(basename "$log" .log)
    if tmux has-session -t "$s" 2>/dev/null; then head -1 "$log" | grep -o 'gpu=[0-9]*' | cut -d= -f2 || true; fi
done | sort -u)
if [ "$GPU" = auto ]; then
    # never stack onto our own GPUs (a run in its evaluation phase can look idle) or onto EXCLUDE_GPUS
    GPU=$(python3 "$ROOT/scripts/pick_gpu.py" --show --exclude "$(echo $USED ${EXCLUDE_GPUS:-} | tr ' ' ',')") \
        || { echo "no idle GPU right now" >&2; exit 1; }
fi

# Cap how many distinct GPUs our live runs occupy (shared server). Override with MAX_GPUS=n.
MAX_GPUS=${MAX_GPUS:-2}
if ! grep -qx "$GPU" <<< "$USED" && [ "$(grep -c . <<< "$USED")" -ge "$MAX_GPUS" ]; then
    echo "already using $MAX_GPUS GPU(s): $(echo $USED) -- not launching $NAME" >&2; exit 2
fi

mkdir -p "$ROOT/runs"
LOG="$ROOT/runs/$NAME.log"
CMD="source $ROOT/scripts/env.sh && cd $ROOT/$DATASET_DIR && \
echo \"[\$(date +%F_%T)] host=\$(hostname) gpu=$GPU args: $*\" && \
CUDA_VISIBLE_DEVICES=$GPU python -u -W ignore evaluate.py --gpu=0 $* ; \
echo \"[\$(date +%F_%T)] exit code \$?\""
tmux new-session -d -s "$NAME" "bash -c '$CMD' 2>&1 | tee -a $LOG"
echo "launched $NAME on GPU $GPU -> $LOG"
