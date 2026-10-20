#!/bin/bash
# Fetch data/state_traces.json from the FLASH repository (BSD-2-Clause) into this directory.
set -e
cd "$(dirname "$0")"
if [ ! -f state_traces.json ]; then
  git clone --depth 1 https://github.com/PKU-Chengxu/FLASH.git _flash
  cp _flash/data/state_traces.json .
  rm -rf _flash
fi
echo "expected md5 of the copy used in the paper: 5af25a7b0b6f9f84dc8ce66831ad55f6"
md5sum state_traces.json
