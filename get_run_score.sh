#!/bin/bash
# Run scripts/get_run_score.py

# set common-sense Bash options
set -o errexit -o pipefail -o noclobber -o nounset

export XLA_PYTHON_CLIENT_PREALLOCATE=false

N_RUNS=$1
EPOCH=$2
TYPE=$3
BATCH_SZ=$((50000/1000))
# EP_SIZE=390
# STEP=EPOCH * EP_SIZE
# Compute step:
STEP=$(( 390 * $EPOCH))

# /usr/bin/python3 /data_diet/scripts/get_run_score.py . baseline 0 $STEP $BATCH_SZ $TYPE &
# Repeat for 0..($N_RUNS-1)
for i in $(seq 0 $(($N_RUNS-1)))
do
    /usr/bin/python3 /data_diet/scripts/get_run_score.py . baseline $i $STEP $BATCH_SZ $TYPE &

    # wait every 1 jobs
    if [ $((i % 1)) -eq 0 ]
    then
        wait
    fi
done

wait
