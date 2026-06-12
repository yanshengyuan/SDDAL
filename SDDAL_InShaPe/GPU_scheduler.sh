#!/bin/bash
set -euo pipefail

LOCKFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/GPU_scheduler.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "Another GPU_scheduler.sh is already running. Exit."
  exit 1
fi

# ==========================================================
# SDDAL GPU Scheduler Daemon (single file, same directory)
#
# Controls:
#   - SDDAL_GPU_scheduled.sh  (your auto-resume pipeline)
#
# Features:
#   - Runs forever, switches GPU by time window:
#       09:30 -> GPU0
#       21:30 -> GPU1
#   - kill + rerun on GPU switch (kills whole process group)
#   - Records the exact "human-checkable" launch command including
#     the REAL start_round into: commands_history.log
#   - Auto-resume start_round priority:
#       (1) Design_${BEAMSHAPE}/progress.state : next_start_round
#       (2) Design_${BEAMSHAPE}/latest_uncertainty : max uncertainty_<round>_*.png + 1
#       (3) default 1
#   - If progress.state is missing, it will AUTO-CREATE a minimal one
#     on first start so future resumes are authoritative.
#
# IMPORTANT FIX (2026-02-27):
#   - Do NOT set CUDA_VISIBLE_DEVICES=$gpu here, because Trainer.py/Scanner.py
#     already receives --gpu $gpu as a PHYSICAL GPU index. Setting
#     CUDA_VISIBLE_DEVICES would remap device ordinals and cause:
#       RuntimeError: CUDA error: invalid device ordinal
#
# Run:
#   chmod +x GPU_scheduler.sh SDDAL_GPU_scheduled.sh
#   nohup bash GPU_scheduler.sh > scheduler.log 2>&1 &
# ==========================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PIPELINE_SCRIPT="SDDAL_GPU_scheduled.sh"

# ========= pipeline args =========
BEAMSHAPE="gaussian"
LR="0.0001"
INIT_SIZE="100"
INIT_ONLY="false"
END_ROUND="1000"
SCANNER_BS="5"
RETRAIN_FREQ="1"
SCAN_ONLY="false"
# ================================

# Switch times
TIME_GPU0="09:30"
TIME_GPU1="21:30"
CHECK_INTERVAL=20

# Local state/log files (in current dir)
PIDFILE="$SCRIPT_DIR/pipeline.pid"
GPUFILE="$SCRIPT_DIR/current_gpu.txt"
LAST_SWITCH="$SCRIPT_DIR/last_switch.txt"
LOGDIR="$SCRIPT_DIR/logs"
CMDLOG="$SCRIPT_DIR/commands_history.log"

mkdir -p "$LOGDIR"

# Conda (delete if not needed)
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="zernike"

log(){ echo "[`date '+%F %T'`] $*"; }

# Convert "HH:MM" -> minutes since midnight (integer)
to_minutes() {
  local t="$1"
  local h m
  IFS=: read -r h m <<< "$t"
  echo $((10#$h * 60 + 10#$m))
}

kill_pipeline(){
  if [ ! -f "$PIDFILE" ]; then
    log "No pipeline running."
    return 0
  fi

  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -z "${pid:-}" ]; then
    rm -f "$PIDFILE"
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PIDFILE"
    return 0
  fi

  log "Stopping pipeline PGID=$pid (TERM)..."
  kill -TERM "-$pid" 2>/dev/null || true
  sleep 10

  if kill -0 "$pid" 2>/dev/null; then
    log "Force killing PGID=$pid (KILL)..."
    kill -KILL "-$pid" 2>/dev/null || true
  fi

  rm -f "$PIDFILE"
  log "Pipeline stopped."
}

# ---------- infer resume start_round ----------
# Priority:
#   1) progress.state next_start_round
#   2) latest_uncertainty max round + 1
#   3) 1
get_resume_start_round(){
  local pf="$SCRIPT_DIR/Design_${BEAMSHAPE}/progress.state"
  local sr=""

  # 1) Prefer progress.state if available
  if [ -f "$pf" ]; then
    local val
    val="$(grep -E '^next_start_round=' "$pf" | tail -n 1 | cut -d'=' -f2- || true)"
    if [[ "${val:-}" =~ ^[0-9]+$ ]]; then
      sr="$val"
    fi
  fi

  # 2) Fallback: infer from latest_uncertainty filenames
  #    Supports:
  #      uncertainty_<round>_<k>.png
  #      uncertainty_<round>.png
  if [ -z "${sr:-}" ]; then
    local udir="$SCRIPT_DIR/Design_${BEAMSHAPE}/latest_uncertainty"
    if [ -d "$udir" ]; then
      local max_round=""
      max_round="$(ls -1 "$udir"/uncertainty_*.png 2>/dev/null \
        | sed -nE 's|.*/uncertainty_([0-9]+)(_[0-9]+)?\.png|\1|p' \
        | sort -n \
        | tail -n 1 || true)"

      if [[ "${max_round:-}" =~ ^[0-9]+$ ]]; then
        sr="$((max_round + 1))"
      fi
    fi
  fi

  # 3) Default
  if [ -z "${sr:-}" ]; then
    sr="1"
  fi

  echo "$sr"
}

# ---------- auto-create minimal progress.state if missing ----------
ensure_progress_state_exists(){
  local start_round="$1"
  local pf="$SCRIPT_DIR/Design_${BEAMSHAPE}/progress.state"

  if [ -f "$pf" ]; then
    return 0
  fi

  mkdir -p "$SCRIPT_DIR/Design_${BEAMSHAPE}"

  local last_finished=$((start_round - 1))
  if [ "$last_finished" -lt 0 ]; then last_finished=0; fi

  cat > "$pf" << EOF
beamshape=${BEAMSHAPE}
init_size=${INIT_SIZE}
scanner_batch_size=${SCANNER_BS}
last_finished_round=${last_finished}
next_start_round=${start_round}
total_batches_generated=${last_finished}
timestamp=$(date +"%F %T")
EOF

  log "Created minimal progress.state at $pf (next_start_round=$start_round)"
}

append_command_log(){
  local gpu="$1"
  local start_round="$2"
  local log_file="$3"

  local cmd
  cmd="bash $PIPELINE_SCRIPT $BEAMSHAPE $LR $INIT_SIZE $INIT_ONLY $start_round $END_ROUND $gpu $SCANNER_BS $RETRAIN_FREQ $SCAN_ONLY"

  {
    echo "============================================================"
    echo "[`date '+%F %T'`] GPU switch/start"
    echo "  desired_gpu=$gpu"
    echo "  start_round(recorded)=$start_round"
    echo "  log_file=$log_file"
    echo "  cmd: $cmd"
  } >> "$CMDLOG"
}

start_pipeline(){
  local gpu="$1"
  cd "$SCRIPT_DIR"

  if [ ! -f "$SCRIPT_DIR/$PIPELINE_SCRIPT" ]; then
    log "ERROR: $PIPELINE_SCRIPT not found in $SCRIPT_DIR"
    exit 1
  fi

  local start_round
  start_round="$(get_resume_start_round)"

  ensure_progress_state_exists "$start_round"

  local logfile="$LOGDIR/run_gpu${gpu}_$(date +%F_%H%M%S).log"

  append_command_log "$gpu" "$start_round" "$logfile"

  log "Starting on GPU$gpu (resume start_round=$start_round), log=$logfile"

  # NOTE: No CUDA_VISIBLE_DEVICES here. We rely on --gpu $gpu as PHYSICAL index.
  setsid bash -lc "
    source \"$CONDA_SH\"
    conda activate \"$CONDA_ENV\"

    cd \"$SCRIPT_DIR\"

    bash \"$PIPELINE_SCRIPT\" \
      \"$BEAMSHAPE\" \
      \"$LR\" \
      \"$INIT_SIZE\" \
      \"$INIT_ONLY\" \
      9999 \
      \"$END_ROUND\" \
      \"$gpu\" \
      \"$SCANNER_BS\" \
      \"$RETRAIN_FREQ\" \
      \"$SCAN_ONLY\"
  " > "$logfile" 2>&1 &

  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "$gpu" > "$GPUFILE"
  log "Started PID/PGID=$pid"
}

desired_gpu(){
  local now_hm nowm g0 g1
  now_hm="$(date +%H:%M)"
  nowm="$(to_minutes "$now_hm")"
  g0="$(to_minutes "$TIME_GPU0")"
  g1="$(to_minutes "$TIME_GPU1")"

  # GPU1 from TIME_GPU1 -> next day TIME_GPU0 (crosses midnight)
  if (( nowm >= g1 || nowm < g0 )); then
    echo 1
  else
    echo 0
  fi
}

need_switch(){
  local key
  key="$(date +%F_%H:%M)"
  if [ -f "$LAST_SWITCH" ] && [ "$(cat "$LAST_SWITCH" 2>/dev/null || true)" = "$key" ]; then
    return 1
  fi
  return 0
}

mark_switch(){
  date +%F_%H:%M > "$LAST_SWITCH"
}

current_gpu(){
  [ -f "$GPUFILE" ] && cat "$GPUFILE" || echo none
}

log "Scheduler started in $SCRIPT_DIR"
log "Switch rule: ${TIME_GPU0}->GPU0, ${TIME_GPU1}->GPU1"
log "Command history log: $CMDLOG"

# Start immediately on correct GPU for current time,
# BUT avoid double-start if pipeline already running.
gpu_now="$(desired_gpu)"

if [ -f "$PIDFILE" ]; then
  pid_existing="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ "${pid_existing:-}" =~ ^[0-9]+$ ]] && kill -0 "$pid_existing" 2>/dev/null; then
    log "Pipeline already running (PID/PGID=$pid_existing). Will not start a new one."
    if [ ! -f "$GPUFILE" ]; then
      echo "$gpu_now" > "$GPUFILE"
    fi
  else
    start_pipeline "$gpu_now"
  fi
else
  start_pipeline "$gpu_now"
fi

mark_switch

while true; do
  desired="$(desired_gpu)"
  current="$(current_gpu)"

  if [ "$desired" != "$current" ]; then
    if need_switch; then
      log "Switch GPU $current -> $desired"
      kill_pipeline
      start_pipeline "$desired"
      mark_switch
    fi
  fi

  sleep "$CHECK_INTERVAL"
done