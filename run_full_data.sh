#!/bin/bash
# Run scripts/run_full_data.py

# set common-sense Bash options
set -o errexit -o pipefail -o noclobber -o nounset

export XLA_PYTHON_CLIENT_PREALLOCATE=false

#/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 0
/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 1 &
/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 2 &

wait

/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 3 &
/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 4 &

# Wait for all bg jobs to finish
wait

/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 5 &
/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 6 &

wait

/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 7 &
/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 8 &

wait

/usr/bin/python3 /data_diet/scripts/run_full_data.py . baseline 9

