#!/bin/bash

# Usage: bash TrainSet_curve.sh <beamshape> <learning_rate>
# Example: bash TrainSet_curve.sh ring 0.0002

BEAMSHAPE="$1"      # e.g., ring
LR="$2"             # e.g., 0.0002

if [ -z "$BEAMSHAPE" ] || [ -z "$LR" ]; then
    echo "Usage: $0 <beamshape> <learning_rate>"
    exit 1
fi

# List of num_samples
samples=(500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

idx=0
for n in "${samples[@]}"; do
    # Assign GPU: first 5 jobs go to gpu0, the rest go to gpu1
    if [ "$idx" -lt 7 ]; then
        gpu=0
    else
        gpu=1
    fi
    idx=$((idx + 1))

    # Automatically determine val_vis_dir based on beamshape and num_samples
    VAL_VIS_DIR="${BEAMSHAPE}_${n}"

    out_file="${BEAMSHAPE}_${n}.txt"
    pth_name="${BEAMSHAPE}_${n}.pth.tar"

    echo "Launching training for beamshape=${BEAMSHAPE}, num_samples=${n} on GPU $gpu"

    nohup python3 train_unet.py \
        --data "../InShaPe_dataset/dense${BEAMSHAPE}30k_pre" \
        --epochs 30 \
        --batch_size 2 \
        --gpu "$gpu" \
        --lr "$LR" \
        --step_size 2 \
        --seed 123 \
        --pth_name "$pth_name" \
        --val_vis_path "$VAL_VIS_DIR" \
        --num_samples "$n" \
        > "$out_file" 2>&1 &
done

echo "All training jobs launched."
