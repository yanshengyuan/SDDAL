#!/bin/bash

# ==========================================
# Usage:
#   bash SDDAL.sh <beamshape> <lr> <initial_size> <fix_init?> <init_only?> <start_round> <end_round> <gpu> <scanner_batch_size> <retrain_frequency> <scan_only?> <seed> <init_seed>
#
# Argument meaning:
#   fix_init:
#       true  -> use external fixed initial set (only when initialization is needed)
#       false -> generate initial set via Initializer.py
#
#   init_only:
#       true  -> only initialize + train once, then exit
#       false -> run full SDDAL pipeline
#
#   scan_only:
#       true  -> skip training, only run Scanner each round
#       false -> normal training + scanning
#
#   start_round:
#       =1    -> fresh start (initialization may be triggered)
#       >1    -> resume mode (existing training_set will NOT be modified)
#
# ==========================================
# Examples:
#
# 1) Regular full pipeline
#    bash SDDAL.sh rec 0.0002 100 false false 1 200 0 5 1 false 12345 321
#
# 2) Full pipeline with fixed initial set
#    bash SDDAL.sh rec 0.0002 100 true false 1 200 0 5 1 false 12345 321
#
# 3) Init-only with generated initial set
#    bash SDDAL.sh rec 0.0002 100 false true 1 200 0 5 1 false 12345 321
#
# 4) Init-only with fixed initial set
#    bash SDDAL.sh rec 0.0002 100 true true 1 200 0 5 1 false 12345 321
#
# 5) Resume full pipeline
#    bash SDDAL.sh rec 0.0002 100 true false 101 200 0 5 1 false 12345 321
#
# 6) Resume full pipeline (fix_init=false also behaves the same in resume mode)
#    bash SDDAL.sh rec 0.0002 100 false false 101 200 0 5 1 false 12345 321
#
# 7) Scan-only from scratch
#    bash SDDAL.sh rec 0.0002 100 false false 1 200 0 5 1 true 12345 321
#
# 8) Resume scan-only
#    bash SDDAL.sh rec 0.0002 100 true false 101 200 0 5 1 true 12345 321
#
# ==========================================

beamshape=${1:-chair}
lr=${2:-0.0002}
init_size=${3:-100}
fix_init=${4:-true}
init_only=${5:-false}
start_round=${6:-1}
end_round=${7:-200}
gpu=${8:-0}
scanner_batch_size=${9:-5}
retrain_freq=${10:-1}
scan_only=${11:-false}
seed=${12:-1}
init_seed=${13:-123}

trainer_batch_size=2  # fixed for Trainer.py
base_path="Design_${beamshape}"
exp_path="Design_${beamshape}_${seed}"

# --- GPU memory waiting config for Scanner.py ---
required_free_mem_mb=8000
check_interval_sec=10

echo "========================================="
echo " Simulation-Driven Differentiable Active Learning (SDDAL) Pipeline"
echo " Beamshape            : ${beamshape}"
echo " Learning rate        : ${lr}"
echo " Rounds               : ${start_round} → ${end_round}"
echo " GPU                  : ${gpu}"
echo " Trainer batch size   : ${trainer_batch_size}"
echo " Scanner batch size   : ${scanner_batch_size}"
echo " Initial set size     : ${init_size}"
echo " Use fixed initial set: ${fix_init}"
echo " Retrain frequency    : ${retrain_freq}"
echo " Scan only?           : ${scan_only}"
echo " Init only?           : ${init_only}"
echo " Master seed          : ${seed}"
echo " Initial set seed     : ${init_seed}"
echo " Base path            : ${base_path}"
echo " Experiment path      : ${exp_path}"
echo " Required free GPU memory before Scanner.py : ${required_free_mem_mb} MB"
echo " GPU memory check interval                 : ${check_interval_sec} s"
echo "========================================="

# ---------------------------------------------------------
# Wait until the target GPU has enough free memory
# ---------------------------------------------------------
wait_for_gpu_memory() {
    local gpu_id="$1"
    local required_mb="$2"
    local interval_sec="$3"

    echo "-----------------------------------------"
    echo " Checking free GPU memory before Scanner.py"
    echo " GPU index          : ${gpu_id}"
    echo " Required free mem  : ${required_mb} MB"
    echo "-----------------------------------------"

    while true; do
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "ERROR: nvidia-smi not found. Cannot check GPU memory."
            exit 1
        fi

        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | head -n 1 | tr -d '[:space:]')

        if ! [[ "${free_mem}" =~ ^[0-9]+$ ]]; then
            echo "$(date '+%F %T')  WARNING: Failed to read free GPU memory on GPU ${gpu_id}. Retry in ${interval_sec}s..."
            sleep "${interval_sec}"
            continue
        fi

        echo "$(date '+%F %T')  GPU ${gpu_id} free memory: ${free_mem} MB"

        if [ "${free_mem}" -ge "${required_mb}" ]; then
            echo "$(date '+%F %T')  Enough free memory detected. Proceeding to run Scanner.py."
            break
        else
            echo "$(date '+%F %T')  Not enough free memory (< ${required_mb} MB). Waiting ${interval_sec}s..."
            sleep "${interval_sec}"
        fi
    done
}

# --- Create experiment folder from base folder if needed ---
echo "-----------------------------------------"
echo " Checking experiment folder: ${exp_path}"
echo "-----------------------------------------"

if [ -d "${exp_path}" ]; then
    echo "  Folder exists -> skip copying."
else
    echo "  Folder does NOT exist -> creating from ${base_path}"
    if [ -d "${base_path}" ]; then
        cp -r "${base_path}" "${exp_path}"
        echo "  Copied ${base_path} -> ${exp_path}"
    else
        echo "  ERROR: Base folder ${base_path} does not exist!"
        exit 1
    fi
fi

echo "-----------------------------------------"

# =========================================================
# Case 1: init_only=true
# - if fix_init=true  -> copy fixed initial set
# - if fix_init=false -> run Initializer.py
# Then train once and exit.
# =========================================================
if [ "${init_only}" = true ]; then
    echo "------------------------------"
    echo "  init_only=true"
    echo "------------------------------"

    if [ "${fix_init}" = true ]; then
        echo "  fix_init=true -> Using external fixed initial set"

        if [ ! -d "../initial_sets/${beamshape}" ]; then
            echo "  ERROR: Fixed initial set folder does not exist:"
            echo "         ../initial_sets/${beamshape}"
            exit 1
        fi

        mkdir -p "${exp_path}/training_set"
        rm -rf "${exp_path}/training_set/"*
        cp -r ../initial_sets/${beamshape}/* "${exp_path}/training_set/"
		mkdir -p "${exp_path}/training_set/init_zernikes"

        echo "  Fixed initial set copied from:"
        echo "    ../initial_sets/${beamshape}/"
        echo "  to:"
        echo "    ${exp_path}/training_set/"
    else
        echo "  Running Initializer.py..."
        python3 Initializer.py \
            --beamshape ${beamshape} \
            --gpu ${gpu} \
            --init_size ${init_size} \
            --vis_path "${exp_path}" \
            --rand_seed ${init_seed}
    fi

    echo "  Training model on initial set..."
	init_train_seed=$(( seed * 1000000 + 3 ))
    python3 Trainer.py \
        --train_data "${exp_path}" \
        --epochs 15 \
        --batch_size ${trainer_batch_size} \
        --gpu ${gpu} \
        --lr ${lr} \
        --step_size 2 \
        --seed 123 \
        --pth_name "${exp_path}/models/QuantUNetT_${beamshape}"

    echo "------------------------------"
    echo "  init_only pipeline finished."
    echo "------------------------------"
    exit 0
fi

# =========================================================
# Case 2: normal mode (init_only=false)
#
# Rules:
# - if start_round > 1:
#       true resume mode, NEVER touch training_set,
#       regardless of fix_init true/false
#
# - if start_round == 1:
#       if scan_only=true:
#           skip initialization
#       else:
#           if fix_init=true:
#               copy fixed initial set
#           else:
#               run Initializer.py
# =========================================================
if [ "${start_round}" -gt 1 ]; then
    echo "------------------------------"
    echo "  Resume mode detected (start_round=${start_round})"
    echo "  Existing training_set will be kept unchanged."
    echo "  Initializer/fixed-initial-set copy both skipped."
    echo "------------------------------"

elif [ "${scan_only}" = true ]; then
    echo "------------------------------"
    echo "  scan_only=true -> Skipping initialization at round 1"
    echo "------------------------------"

else
    # Here: init_only=false, start_round==1, scan_only=false
    if [ "${fix_init}" = true ]; then
        echo "------------------------------"
        echo "  start_round=1 and fix_init=true"
        echo "  Using external fixed initial set instead of Initializer.py"
        echo "------------------------------"

        if [ ! -d "../initial_sets/${beamshape}" ]; then
            echo "  ERROR: Fixed initial set folder does not exist:"
            echo "         ../initial_sets/${beamshape}"
            exit 1
        fi

        mkdir -p "${exp_path}/training_set"
        rm -rf "${exp_path}/training_set/"*
        cp -r ../initial_sets/${beamshape}/* "${exp_path}/training_set/"
		mkdir -p "${exp_path}/training_set/init_zernikes"

        echo "  Fixed initial set copied from:"
        echo "    ../initial_sets/${beamshape}/"
        echo "  to:"
        echo "    ${exp_path}/training_set/"
    else
        echo "------------------------------"
        echo "  start_round=1 and fix_init=false"
        echo "  Running Initializer.py"
        echo "------------------------------"

        python3 Initializer.py \
            --beamshape ${beamshape} \
            --gpu ${gpu} \
            --init_size ${init_size} \
            --vis_path "${exp_path}" \
            --rand_seed ${init_seed}
    fi
fi

# =========================================================
# Main round loop
# =========================================================
for ((round_sampling=${start_round}; round_sampling<=${end_round}; round_sampling++))
do
    echo "------------------------------"
    echo "  Starting Round ${round_sampling}"
    echo "------------------------------"
	
	scanner_init_seed=$(( seed * 1000000 + round_sampling * 2 ))
	train_seed=$(( seed * 1000000 + round_sampling * 2 + 3 ))

    echo "  Scanner init_seed   : ${scanner_init_seed}"
    echo "  Scanner reset_seed  : ${reset_seed}"
	echo "  Trainer seed  : ${train_seed}"

    # Re-train only when round index matches frequency
    # If scan_only=true -> skip training completely
    if [ "${scan_only}" = false ]; then
        if (( (round_sampling-1) % retrain_freq == 0 )); then
            echo "------------------------------"
            echo "  Re-training model at round ${round_sampling}"
            echo "  (Training happens every ${retrain_freq} scans)"
            echo "------------------------------"

            python3 Trainer.py \
                --train_data "${exp_path}" \
                --epochs 15 \
                --batch_size ${trainer_batch_size} \
                --gpu ${gpu} \
                --lr ${lr} \
                --step_size 2 \
                --seed 123 \
                --pth_name "${exp_path}/models/QuantUNetT_${beamshape}"
        else
            echo "  Skipping training at this round (waiting for next frequency point)"
        fi
    else
        echo "  scan_only=true -> training skipped."
    fi

    # Round-specific seeds for Scanner.py
    # Guarantees:
    # 1) different rounds use different seeds
    # 2) within the same round: init_seed != reset_seed
    # 3) changing master seed changes the whole seed sequence

    # Wait until GPU has enough free memory before running Scanner.py
    wait_for_gpu_memory "${gpu}" "${required_free_mem_mb}" "${check_interval_sec}"

    # Run Scanner every round
    python3 Scanner.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --batch_size ${scanner_batch_size} \
        --pth_name QuantUNetT_${beamshape} \
        --round_sampling ${round_sampling} \
        --vis_path "${exp_path}" \
        --init_seed ${scanner_init_seed}

done

# =========================================================
# Zernike coefficients statistics
# =========================================================
echo "========================================="
echo "   Zernike coefficient statistics in progress..."
echo "========================================="
python3 zernike_statistics.py --beamshape ${beamshape} --init_size ${init_size}

echo "========================================="
echo "   SDDAL (Simulation-Driven Differentiable Active Learning) Pipeline Completed Successfully!"
echo "========================================="