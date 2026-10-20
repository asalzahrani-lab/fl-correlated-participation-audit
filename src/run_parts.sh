#!/bin/bash
# Supervisor: runs the missing parts of one step20b job sequentially, then combines.
# usage: bash run_parts.sh markov 12 250000 [extra args for step20b: fine T round_min win_days stride]
# The part files are skipped when present, so the script can be relaunched after a container restart.
cd "$(dirname "$0")"
which=$1; nparts=$2; npart=$3; shift 3
extra="$@"
suf=""
for a in $extra; do :; done
for ((k=0; k<nparts; k++)); do
  python3 step20b_parts.py sample $which $k $npart $extra >> step20b_${which}_$(echo $extra | tr ' ' '_').log 2>&1
done
python3 step20b_parts.py combine $which $extra > step20b_combine_${which}_$(echo $extra | tr ' ' '_').log 2>&1
echo done >> step20b_combine_${which}_$(echo $extra | tr ' ' '_').log
