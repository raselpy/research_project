#!/usr/bin/env bash
# Resumable parallel runner for src.training.cv experiments.
#
#   ./run_experiments.sh              # run everything not yet finished
#   ./run_experiments.sh --status     # show what is done / pending
#   ./run_experiments.sh --reset ablation_region   # forget one experiment
#   ./run_experiments.sh --reset all  # forget everything
#
# Env overrides:
#   MAX_PARALLEL=3 ./run_experiments.sh
#   GPUS="0,1"     ./run_experiments.sh     # round-robin experiments over GPUs

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

COMMON_ARGS=(
  --num-folds 5
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

done_marker() { echo "$STATE_DIR/$1.done"; }

show_status() {
  printf '%-28s %s\n' "EXPERIMENT" "STATE"
  for exp in "${EXPERIMENTS[@]}"; do
    if [[ -f "$(done_marker "$exp")" ]]; then
      printf '%-28s %s\n' "$exp" "done"
    else
      printf '%-28s %s\n' "$exp" "pending"
    fi
  done
}

case "${1:-}" in
  --status)
    show_status; exit 0 ;;
  --reset)
    target="${2:-}"
    if [[ "$target" == "all" ]]; then
      rm -f "$STATE_DIR"/*.done
      echo "cleared all markers"
    elif [[ -n "$target" ]]; then
      rm -f "$(done_marker "$target")"
      echo "cleared marker for $target"
    else
      echo "usage: $0 --reset <experiment|all>" >&2; exit 1
    fi
    exit 0 ;;
esac

run_one() {
  local exp="$1" gpu="$2"
  local log="$LOG_DIR/$exp.log"

  echo "[$(date '+%F %T')] START $exp${gpu:+ (GPU $gpu)}"
  {
    echo "=== $exp started $(date '+%F %T') ==="
    if [[ -n "$gpu" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" python -m src.training.cv --experiment "$exp" "${COMMON_ARGS[@]}"
    else
      python -m src.training.cv --experiment "$exp" "${COMMON_ARGS[@]}"
    fi
  } >>"$log" 2>&1

  local rc=$?
  if [[ $rc -eq 0 ]]; then
    date '+%F %T' > "$(done_marker "$exp")"
    echo "[$(date '+%F %T')] DONE  $exp"
  else
    echo "[$(date '+%F %T')] FAIL  $exp (exit $rc) -- see $log"
  fi
  return $rc
}

# Build the pending list
pending=()
for exp in "${EXPERIMENTS[@]}"; do
  if [[ -f "$(done_marker "$exp")" ]]; then
    echo "skip (already done): $exp"
  else
    pending+=("$exp")
  fi
done

if [[ ${#pending[@]} -eq 0 ]]; then
  echo "Nothing to do - all experiments finished."
  exit 0
fi

IFS=',' read -r -a gpu_list <<< "$GPUS"
gpu_count=${#gpu_list[@]}
[[ -z "$GPUS" ]] && gpu_count=0

i=0
for exp in "${pending[@]}"; do
  # throttle to MAX_PARALLEL concurrent jobs
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
    wait -n 2>/dev/null || sleep 5
  done

  gpu=""
  if [[ $gpu_count -gt 0 ]]; then
    gpu="${gpu_list[$(( i % gpu_count ))]}"
  fi

  run_one "$exp" "$gpu" &
  i=$((i + 1))
  sleep 2   # stagger starts so the runs don't collide on init
done

wait

echo
echo "===== summary ====="
show_status

for exp in "${pending[@]}"; do
  [[ -f "$(done_marker "$exp")" ]] || exit 1
done