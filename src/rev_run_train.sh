#!/bin/bash
# The 200 training runs of the revision (Table VI of the full version). rev_train.py is resumable and replays the
# random stream of earlier repetitions, so this single script reproduces the two batches that were run in the paper
# (first 10/10/8/10/4 repetitions, then the rest).
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
python3 rev_train.py base_orth 50   >> rev_train.log 2>&1
python3 rev_train.py medium 40      >> rev_train.log 2>&1
python3 rev_train.py ratio_1to10 40 >> rev_train.log 2>&1
python3 rev_train.py ratio_1to2 30  >> rev_train.log 2>&1
python3 rev_train.py medium_iid 20  >> rev_train.log 2>&1
python3 rev_train.py base_rand 20   >> rev_train.log 2>&1
echo ALLDONE >> rev_train.log
