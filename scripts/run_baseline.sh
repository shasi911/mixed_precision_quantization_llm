#!/usr/bin/env bash
# Run FP16 baseline – no quantization applied
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/quantization_config.yaml"

echo "=============================="
echo " Baseline: FP16 (no quant)"
echo "=============================="

cd "$PROJECT_DIR"
python main.py --config "$CONFIG" --mode baseline
