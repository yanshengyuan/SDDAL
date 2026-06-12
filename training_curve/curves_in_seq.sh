#!/bin/bash

# Usage:
#   bash run_all_beams.sh <data_root>
# Example:
#   bash run_all_beams.sh ../SDDAL_InShaPe_randinit_syncUNet_varseed

DATA_ROOT="$1"
BEAMSHAPE="$2"

if [ -z "$DATA_ROOT" ]; then
    echo "Usage: $0 <data_root>"
    exit 1
fi

wait_until_beam_done () {
    local beam="$1"

    while pgrep -f "python3 retrain.py .*Design_${beam}" > /dev/null; do
        echo "$(date '+%F %T')  Waiting for ${beam} jobs to finish..."
        sleep 20
    done
}

run_one_beam () {
    local beam="$1"
    local lr="$2"
	local seed="$3"

    echo "=================================================="
    echo "$(date '+%F %T')  Launching beam=${beam}, lr=${lr}"
    echo "=================================================="

    bash TrainSet_curve_sddal.sh "$beam" "$lr" "$DATA_ROOT" "$seed"

    echo "$(date '+%F %T')  All ${beam} jobs launched. Now waiting..."
    wait_until_beam_done "$beam"

    echo "$(date '+%F %T')  ${beam} jobs finished. Running eval_all.sh..."

    pushd "$beam" > /dev/null || {
        echo "$(date '+%F %T')  Failed to enter folder: $beam"
        return 1
    }

    bash eval_all.sh

    popd > /dev/null || {
        echo "$(date '+%F %T')  Failed to return to project root"
        return 1
    }

    echo "$(date '+%F %T')  ${beam} completely finished."
    echo
}

run_one_beam "${BEAMSHAPE}_1" 0.0002 9999
run_one_beam "${BEAMSHAPE}_2" 0.0002 9999
run_one_beam "${BEAMSHAPE}_3" 0.0002 9999
run_one_beam "${BEAMSHAPE}_4" 0.0002 9999
run_one_beam "${BEAMSHAPE}_5" 0.0002 9999

echo "$(date '+%F %T')  All beamshapes finished."