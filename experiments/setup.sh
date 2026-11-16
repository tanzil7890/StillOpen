#!/usr/bin/env bash
# One-time setup for everything under experiments/. Safe to run again: each step is skipped when already done.
#   bash experiments/setup.sh
# Creates experiments/.venv, and downloads and parses the BPI Challenge 2012 log for the pilot. The logs of the
# censoring study are downloaded by censoring/prepare.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RAW=data/bpi12_raw/BPI_Challenge_2012.xes.gz

[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -q -r requirements.txt

if [ ! -f "$RAW" ]; then
  mkdir -p data/bpi12_raw
  curl -L -o "$RAW" "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893"
fi
sum=$(md5 -q "$RAW" 2>/dev/null || md5sum "$RAW" | cut -d' ' -f1)
[ "$sum" = 74c7ba9aba85bfcb181a22c9d565e5b5 ] || { echo "dataset checksum mismatch: $sum"; exit 1; }

mkdir -p data/derived
[ -f data/derived/bpi12.pkl ] || .venv/bin/python -I pilot/parse_bpi12.py "$RAW" data/derived/bpi12.pkl
echo "setup done"
