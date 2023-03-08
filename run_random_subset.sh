#!/bin/bash
# Run scripts/run_full_data.py

# set common-sense Bash options
set -o errexit -o pipefail -o noclobber -o nounset

export XLA_PYTHON_CLIENT_PREALLOCATE=false

/usr/bin/python3 /data_diet/scripts/run_random_subset.py . baseline $1 0 &
/usr/bin/python3 /data_diet/scripts/run_random_subset.py . baseline $1 1 &

wait

/usr/bin/python3 /data_diet/scripts/run_random_subset.py . baseline $1 2 &
/usr/bin/python3 /data_diet/scripts/run_random_subset.py . baseline $1 3 &

wait

/usr/bin/python3 /data_diet/scripts/run_random_subset.py . baseline $1 4 &
/usr/bin/python3 /data_diet/scripts/run_random_subset.py . baseline $1 5 &

