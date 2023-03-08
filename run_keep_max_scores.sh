#!/bin/bash
# Run scripts/run_keep_max_scores.py

# set common-sense Bash options
set -o errexit -o pipefail -o noclobber -o nounset

export XLA_PYTHON_CLIENT_PREALLOCATE=false

TARGET=/data_diet/scripts/run_keep_max_scores.py
N_RUNS=$1
TYPE=$2
SUBSET_SIZE=$3
EPOCH=$4
STEP=$(( 390 * $EPOCH))

CMDLINE="$TARGET . keep_max_${TYPE}_${EPOCH} ./exps/baseline/$TYPE/ckpt_$STEP.npy $SUBSET_SIZE"

# Repeat for 0..($N_RUNS-1)
for i in $(seq 0 $(($N_RUNS-1)))
do
    /usr/bin/python3 $CMDLINE $i &

    # wait every 2 jobs
    if [ $((i % 2)) -eq 1 ]
    then
        wait
    fi
done