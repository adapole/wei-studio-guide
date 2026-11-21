#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
case "${1:-}" in
  prepare)
    sha256sum -c SHA256SUMS
    test ! -e inputs || { echo "inputs already exists" >&2; exit 1; }
    model="$root/../models/orca_sim"
    test "$(git -C "$model" rev-parse HEAD)" = 05e00dafd0a6eb2186d69d4e351c252a7bcffadd
    mkdir inputs
    tar xzf inputs.tar.gz -C inputs
    tar xzf orca-runtime.tar.gz -C inputs
    unzip -q checkpoint.zip -d inputs
    python prepare_model.py "$model" inputs/artifacts/orca_sim
    ln -s "$root/scripts" inputs/scripts
    ;;
  train)
    python scripts/orca-table-train-versioned.py --workspace "$root/inputs" --bundle "$root/inputs/admitted" --dataset "$root/inputs/dataset" --output "$root/outputs/train" --seed 7 --steps 20000 --batch-size 32 --normalize-environment
    ;;
  evaluate)
    checkpoint="${2:-$root/inputs/pretrained_model}"
    output="${3:-$root/outputs/validation}"
    python scripts/orca-table-evaluate.py --workspace "$root/inputs" --bundle "$root/inputs/admitted" --checkpoint "$checkpoint" --output "$output" --split validation --startup-seconds 0.2
    python -c 'import json,sys; r=json.load(open(sys.argv[1])); print("Held-out passes:",r["successes"],"/",len(r["episodes"])); sys.exit(0 if r["complete"] and r["successes"]==len(r["episodes"])==2 else 1)' "$output/report.json"
    ;;
  *) echo "Usage: bash run.sh prepare|train|evaluate [checkpoint] [new-output-dir]" >&2; exit 2 ;;
esac
