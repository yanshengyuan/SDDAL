#!/bin/bash

PREFIX="$1"      # e.g., ring
USER_HOST="syan@131.155.70.184:/data-3/users/syan/NED_inshape/inshape"
MAX_JOBS=6       # maximum concurrent SCP transfers

samples=(500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

job_count=0

for n in "${samples[@]}"; do
    folder="${PREFIX}_${n}"
    if [ -d "$folder" ]; then
        echo "Transferring $folder..."
        scp -r "$folder" "$USER_HOST/$folder" &  # run in background
        ((job_count++))
        
        # wait if MAX_JOBS are running
        if (( job_count >= MAX_JOBS )); then
            wait -n  # waits for any background job to finish
            ((job_count--))
        fi
    else
        echo "Folder $folder does not exist yet. Skipping."
    fi
done

# wait for all remaining jobs
wait
echo "All SCP transfers done."
