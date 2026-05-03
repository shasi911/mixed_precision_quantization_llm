#!/usr/bin/env bash
# Full experiment suite:
#   1. FP16 baseline
#   2. Uniform INT4 baseline
#   3. Uniform INT8 baseline
#   4. Layer-aware mixed precision (from YAML config)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/quantization_config.yaml"

cd "$PROJECT_DIR"

echo "=============================="
echo " 1/4  FP16 Baseline"
echo "=============================="
python main.py --config "$CONFIG" --mode baseline

echo ""
echo "=============================="
echo " 2/4  Uniform INT4"
echo "=============================="
python main.py --config "$CONFIG" --mode uniform --uniform_bits 4

echo ""
echo "=============================="
echo " 3/4  Uniform INT8"
echo "=============================="
python main.py --config "$CONFIG" --mode uniform --uniform_bits 8

echo ""
echo "=============================="
echo " 4/4  Layer-Aware Mixed Quant"
echo "=============================="
python main.py --config "$CONFIG" --mode quantize

echo ""
echo "All experiments complete.  Results are in: ${PROJECT_DIR}/results/"
