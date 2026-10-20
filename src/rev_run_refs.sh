#!/bin/bash
# Runs the revision Monte-Carlo references part by part (restart-safe), then combines.
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
run() { law=$1; nparts=$2; n=$3
  for ((k=0; k<nparts; k++)); do python3 rev_refs.py sample $law $k $n >> rev_refs_$law.log 2>&1; done
  python3 rev_refs.py combine $law >> rev_refs_$law.log 2>&1; echo "combined $law" >> rev_refs_$law.log; }
run markov200 20 500000
run timer20 6 500000
run timer5 6 500000
run droptail 6 500000
run trace200 16 500000
echo ALLDONE > rev_refs_done.flag
