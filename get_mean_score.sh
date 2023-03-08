#!/bin/bash
# Run scripts/get_mean_score.py

# set common-sense Bash options
set -o errexit -o pipefail -o noclobber -o nounset

export XLA_PYTHON_CLIENT_PREALLOCATE=false

N_RUNS=$1
EPOCH=$2
TYPE=$3
# EP_SIZE=390
# STEP=EPOCH * EP_SIZE
# Compute step:
STEP=$(( 390 * $EPOCH))

/usr/bin/python3 /data_diet/scripts/get_mean_score.py . baseline $N_RUNS $STEP $TYPE