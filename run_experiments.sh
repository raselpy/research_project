#!/usr/bin/env bash
# Resumable parallel runner for src.training.cv experiments.
# Resume granularity: EXPERIMENT + FOLD (not epoch - see note at bottom of file).
#
#   ./run_experiments.sh              # run everything not yet finished
#   ./run_experiments.sh --status     # show what is done / pending, per fold
#   ./run_experiments.sh --reset ablation_region        # forget one experiment (all its folds)
#   ./run_experiments.sh --reset ablation_region 2       # forget just fold 2 of one experiment
#   ./run_experiments.sh --reset all  # forget everything
#
# Env overrides:
#   MAX_PARALLEL=3 ./run_experiments.sh
#   GPUS="0,1"     ./run_experiments.sh     # round-robin fold-jobs over GPUs

set -uo pipefail

EXPERIMENTS=(
  baseline
  baseline_bs5
  ablation_region
  ablation_dataug
  ablation_dataug_star
  ablation_batchnorm
  ablation_batchdice
  ablation_dataug_star_bn
)

NUM_FOLDS=5

# Passed straight through to run.py as overrides, same as cv.py would.
COMMON_OVERRIDES=(
  training.epochs=10
  training.iterations_per_epoch=20
  training.val_every_n_epochs=1
  logging.use_dagshub=true
)

STATE_DIR="${STATE_DIR:-.run_state}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
GPUS="${GPUS:-}"

mkdir -p "$STATE_DIR" "$LOG_DIR"

# Load DagsHub / other secrets from .env if present, without printing them.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "Loaded environment variables from .env"
else
  echo "No .env file found - DAGSHUB_USERNAME/DAGSHUB_REPO/DAGSHUB_TOKEN must already be set in this shell."
fi

for var in DAGSHUB_USERNAME DAGSHUB_REPO DAGSHUB_TOKEN; do
  if [[ -z "${!var:-}" ]]; then
    echo "WARNING: $var is not set. Runs with logging.use_dagshub=true will fail." >&2
  fi
done

done_marker() { echo "$STATE_DIR/$1-fold$2.done"; }

show_status() {
  printf '%-28s %s\n' "EXPERIMENT" "FOLDS DONE"
  for exp in "${EXPERIMENTS[@]}"; do
    done_count=0
    fold_str=""
    for ((f=0; f<NUM_FOLDS; f++)); do
      if [[ -f "$(done_marker "$exp" "$f")" ]]; then
        done_count=$((done_count + 1))
        fold_str+="✓"
      else
        fold_str+="."
      fi
    done
    printf '%-28s %d/%d  [%s]\n' "$exp" "$done_count" "$NUM_FOLDS" "$fold_str"
  done
}

case "${1:-}" in
  --status)
    show_status; exit 0 ;;
  --reset)
    target="${2:-}"
    fold="${3:-}"
    if [[ "$target" == "all" ]]; then
      rm -f "$STATE_DIR"/*.done
      echo "cleared all markers"
    elif [[ -n "$target" && -n "$fold" ]]; then
      rm -f "$(done_marker "$target" "$fold")"
      echo "cleared marker for $target fold $fold"
    elif [[ -n "$target" ]]; then
      rm -f "$STATE_DIR/$target-fold"*.done
      echo "cleared all fold markers for $target"
    else
      echo "usage: $0 --reset <experiment|all> [fold]" >&2; exit 1
    fi
    exit 0 ;;
esac

run_one_fold() {
  local exp="$1" fold="$2" gpu="$3"
  local log="$LOG_DIR/$exp-fold$fold.log"

  echo "[$(date '+%F %T')] START $exp fold $fold${gpu:+ (GPU $gpu)}"
  {
    echo "=== $exp fold $fold started $(date '+%F %T') ==="
    if [[ -n "$gpu" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" python -m src.training.run \
        "experiment=$exp" "experiment_name=$exp" \
        "dataset.fold=$fold" "dataset.num_folds=$NUM_FOLDS" \
        "${COMMON_OVERRIDES[@]}"
    else
      python -m src.training.run \
        "experiment=$exp" "experiment_name=$exp" \
        "dataset.fold=$fold" "dataset.num_folds=$NUM_FOLDS" \
        "${COMMON_OVERRIDES[@]}"
    fi
  } >>"$log" 2>&1

  local rc=$?
  if [[ $rc -eq 0 ]]; then
    date '+%F %T' > "$(done_marker "$exp" "$fold")"
    echo "[$(date '+%F %T')] DONE  $exp fold $fold"
  else
    echo "[$(date '+%F %T')] FAIL  $exp fold $fold (exit $rc) -- see $log"
  fi
  return $rc
}

# Build the pending list of "exp fold" pairs
pending=()
for exp in "${EXPERIMENTS[@]}"; do
  for ((f=0; f<NUM_FOLDS; f++)); do
    if [[ -f "$(done_marker "$exp" "$f")" ]]; then
      : # already done, skip silently (status view shows it)
    else
      pending+=("$exp $f")
    fi
  done
done

if [[ ${#pending[@]} -eq 0 ]]; then
  echo "Nothing to do - all experiments/folds finished."
  exit 0
fi

echo "Pending: ${#pending[@]} (experiment, fold) jobs"

IFS=',' read -r -a gpu_list <<< "$GPUS"
gpu_count=${#gpu_list[@]}
[[ -z "$GPUS" ]] && gpu_count=0

i=0
for pair in "${pending[@]}"; do
  read -r exp fold <<< "$pair"

  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
    wait -n 2>/dev/null || sleep 5
  done

  gpu=""
  if [[ $gpu_count -gt 0 ]]; then
    gpu="${gpu_list[$(( i % gpu_count ))]}"
  fi

  run_one_fold "$exp" "$fold" "$gpu" &
  i=$((i + 1))
  sleep 2   # stagger starts so the runs don't collide on init
done

wait

echo
echo "===== summary ====="
show_status

for pair in "${pending[@]}"; do
  read -r exp fold <<< "$pair"
  [[ -f "$(done_marker "$exp" "$fold")" ]] || exit 1
done

# --- Note on epoch-level resume ---
# This script resumes at the (experiment, fold) level: if power is lost mid-fold,
# that fold restarts from epoch 0, but every already-finished fold is skipped.
# True epoch-level resume needs src/training/run.py itself to save a checkpoint
# each epoch and accept a flag to resume from it - that's a training-loop change,
# not something this orchestration script can do on its own.