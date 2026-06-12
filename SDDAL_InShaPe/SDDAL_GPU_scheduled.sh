#!/bin/bash
set -euo pipefail

# ==========================================
# Usage (仍保留原参数形状，便于兼容你已有的调用方式):
#   bash SDDAL.sh <beamshape> <lr> <initial_size> <init_only?> <start_round> <end_round> <gpu> <scanner_batch_size> <retrain_frequency> <scan_only?>
#
# IMPORTANT (本自动续跑版本的关键规则):
#   - start_round 会被自动续上（从 progress.state 读取），不再使用手动输入的 start_round
#   - “已生成样本数量(不含 initial)” 就等于 round（batch数量），所以 start_round 本质就是 round index
# ==========================================

beamshape=${1:-chair}
lr=${2:-0.0002}
init_size=${3:-100}
init_only=${4:-false}

# start_round 形参保留，但会被下面的自动续跑逻辑覆盖
start_round_arg=${5:-1}

end_round=${6:-580}
gpu=${7:-0}
scanner_batch_size=${8:-5}
retrain_freq=${9:-1}
scan_only=${10:-false}

trainer_batch_size=2  # fixed for Trainer.py

# ------------------------------
# Progress state file (ALWAYS auto-resume)
# start_round := round index (batches generated excluding initial)
# ------------------------------
PROGRESS_FILE="Design_${beamshape}/progress.state"
mkdir -p "Design_${beamshape}"

# Default: start from round 1
start_round=1

if [ -f "${PROGRESS_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${PROGRESS_FILE}" || true
    if [ -n "${next_start_round:-}" ]; then
        start_round="${next_start_round}"
    fi
fi

# Sanity clamp
if [ "${start_round}" -lt 1 ]; then
    start_round=1
fi

# If already finished
if [ "${start_round}" -gt "${end_round}" ]; then
    echo "========================================="
    echo " SDDAL: Nothing to run."
    echo " beamshape     : ${beamshape}"
    echo " progress file : ${PROGRESS_FILE}"
    echo " start_round   : ${start_round}"
    echo " end_round     : ${end_round}"
    echo "========================================="
    exit 0
fi

echo "========================================="
echo " SDDAL Pipeline (AUTO-RESUME ENABLED)"
echo " Beamshape              : ${beamshape}"
echo " Learning rate          : ${lr}"
echo " GPU                    : ${gpu}"
echo " Trainer batch size     : ${trainer_batch_size}"
echo " Scanner batch size     : ${scanner_batch_size}"
echo " Initial set size       : ${init_size}"
echo " Retrain frequency      : ${retrain_freq}"
echo " Scan only?             : ${scan_only}"
echo " Init only?             : ${init_only}"
echo "-----------------------------------------"
echo " start_round (auto)     : ${start_round}"
echo " end_round              : ${end_round}"
echo " (start_round arg given : ${start_round_arg}  -> IGNORED)"
echo " progress file          : ${PROGRESS_FILE}"
echo "========================================="

# --- Handle init_only=true separately ---
if [ "${init_only}" = true ]; then
    echo "------------------------------"
    echo "  init_only=true → Generate initial training set and train once"
    echo "------------------------------"

    echo "  Running Initializer.py..."
    python3 Initializer.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --init_size ${init_size} \
        --vis_path Design_${beamshape}

    echo "  Training model on initial set..."
    python3 Trainer.py \
        --train_data Design_${beamshape} \
        --epochs 15 \
        --batch_size ${trainer_batch_size} \
        --gpu ${gpu} \
        --lr ${lr} \
        --step_size 2 \
        --seed 123 \
        --pth_name Design_${beamshape}/models/QuantUNetT_${beamshape}

    # init_only 也写一个 progress（可选，但我建议写：表示“尚未开始 round 扫描”）
    cat > "${PROGRESS_FILE}" << EOF
beamshape=${beamshape}
init_size=${init_size}
scanner_batch_size=${scanner_batch_size}
last_finished_round=0
next_start_round=1
total_batches_generated=0
timestamp=$(date +"%F %T")
EOF

    echo "------------------------------"
    echo "  init_only pipeline finished."
    echo "  progress state written: ${PROGRESS_FILE}"
    echo "------------------------------"
    exit 0
fi

# --- Regular behavior (init_only=false) ---
# Run Initializer.py only when:
#   - scan_only=false
#   - and progress indicates we haven't initialized (optional)
#
# 你原脚本逻辑是：start_round==1 才 initializer。
# 但现在 start_round 是自动续跑的：如果你第一次跑，它就是 1，没问题。
# 如果你断点续跑 start_round>1，那肯定不该 initializer。
if [ "${scan_only}" = true ]; then
    echo "------------------------------"
    echo "  scan_only=true → Skipping Initializer.py"
    echo "------------------------------"
elif [ "${start_round}" -eq 1 ]; then
    echo "------------------------------"
    echo "  Running Initializer.py (first run)"
    echo "------------------------------"

    python3 Initializer.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --init_size ${init_size} \
        --vis_path Design_${beamshape}
else
    echo "------------------------------"
    echo "  Skipping Initializer.py (auto-resume from round ${start_round})"
    echo "------------------------------"
fi

# Loop over rounds
for ((round_sampling=${start_round}; round_sampling<=${end_round}; round_sampling++))
do
    echo "------------------------------"
    echo "  Starting Round ${round_sampling}"
    echo "  (This round index == total batches generated excluding initial)"
    echo "------------------------------"

    # Re-train only when round index matches frequency
    # If scan_only=true → skip training completely
    if [ "${scan_only}" = false ]; then
        if (( (round_sampling-1) % retrain_freq == 0 )); then
            echo "------------------------------"
            echo "  Re-training model at round ${round_sampling}"
            echo "  (Training happens every ${retrain_freq} scans)"
            echo "------------------------------"

            python3 Trainer.py \
                --train_data Design_${beamshape} \
                --epochs 15 \
                --batch_size ${trainer_batch_size} \
                --gpu ${gpu} \
                --lr ${lr} \
                --step_size 2 \
                --seed 123 \
                --pth_name Design_${beamshape}/models/QuantUNetT_${beamshape}
        else
            echo "  Skipping training at this round (waiting for next frequency point)"
        fi
    else
        echo "  scan_only=true → training skipped."
    fi

    # Run Scanner every round (adds new samples to the dataset)
    python3 Scanner.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --batch_size ${scanner_batch_size} \
        --pth_name QuantUNetT_${beamshape} \
        --round_sampling ${round_sampling} \
        --vis_path Design_${beamshape}

    # ------------------------------
    # Update progress AFTER a successful scan
    # Here "generated batches excluding initial" == round_sampling
    # So next_start_round is round_sampling + 1
    # ------------------------------
    next_round=$((round_sampling + 1))

    cat > "${PROGRESS_FILE}" << EOF
beamshape=${beamshape}
init_size=${init_size}
scanner_batch_size=${scanner_batch_size}
last_finished_round=${round_sampling}
next_start_round=${next_round}
total_batches_generated=${round_sampling}
timestamp=$(date +"%F %T")
EOF

    echo "  Progress updated: last_finished_round=${round_sampling}, next_start_round=${next_round}"
done

# Zernike coefficients statistics
echo "========================================="
echo "   Zernike coefficient statistics in progress…"
echo "========================================="
python3 zernike_statistics.py --beamshape ${beamshape} --init_size ${init_size}

echo "========================================="
echo "   SDDAL Pipeline Completed Successfully!"
echo "   Progress state: ${PROGRESS_FILE}"
echo "========================================="