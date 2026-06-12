#!/bin/bash

# Usage: bash TrainSet_curve.sh <beamshape> <learning_rate> <data_root>
# Example: bash TrainSet_curve.sh rec 0.0002 ../SDDAL_InShaPe_randinit_syncUNet_varseed

BEAMSHAPE="$1"
LR="$2"
DATA_ROOT="$3"
SEED="$4"

BEAMSHAPE_ARG="${BEAMSHAPE%%_*}"

if [ -z "$BEAMSHAPE" ] || [ -z "$LR" ] || [ -z "$DATA_ROOT" ]; then
    echo "Usage: $0 <beamshape> <learning_rate> <data_root>"
    exit 1
fi

# List of num_samples
samples=(1100 900 600 400 200 1000 800 700 500 300)

idx=0
for n in "${samples[@]}"; do

    # Compute n-100 (only for naming)
    n_minus_100=$((n - 100))

    # First It jobs -> gpu0, rest -> gpu1, set -lt as 999 to run all jobs on gpu0, -lt as 0 to run all jobs on gpu1
    if [ "$idx" -lt 0 ]; then
        gpu=0
    else
        gpu=1
    fi
    idx=$((idx + 1))

    # Use n_minus_100 for naming
	PARENT_DIR="${BEAMSHAPE}"
	VAL_VIS_DIR="${PARENT_DIR}/${BEAMSHAPE}_${n_minus_100}"
	out_file="${BEAMSHAPE}_${n_minus_100}.txt"

	# Create parent directory if it does not exist
	if [ ! -d "$PARENT_DIR" ]; then
		mkdir -p "$PARENT_DIR"
		cp eval_all.sh "$PARENT_DIR/"
		cp FRCM.py "$PARENT_DIR/"
	fi

	echo "Preparing output folder: ${VAL_VIS_DIR}"
	if [ -d "$VAL_VIS_DIR" ]; then
		rm -r "$VAL_VIS_DIR"
	fi
	cp -r gaussian "$VAL_VIS_DIR"

    echo "Launching training for beamshape=${BEAMSHAPE}, num_samples=${n} (named as ${n_minus_100}) on GPU $gpu"

    nohup python3 retrain.py \
        --data "${DATA_ROOT}/Design_${BEAMSHAPE}" \
        --epochs 15 \
        --batch_size 2 \
        --gpu "$gpu" \
        --lr "$LR" \
        --step_size 2 \
        --seed 123 \
        --vis_path "$VAL_VIS_DIR" \
        --num_samples "$n" \
        --init_weight_path "../results/SDDAL_InShaPe_rec/init.pth.tar" \
        > "$out_file" 2>&1 &
done

echo "All training jobs launched."