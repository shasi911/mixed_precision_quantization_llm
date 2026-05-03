#!/usr/bin/env bash
# Layer sensitivity analysis: quantize one layer at a time, measure accuracy drop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/quantization_config.yaml"

DATASET="${1:-boolq}"      # dataset to use for sensitivity probing
SAMPLES="${2:-100}"        # number of samples per probe
OUTPUT="${PROJECT_DIR}/results/sensitivity_${DATASET}.json"

echo "=============================="
echo " Sensitivity Analysis"
echo " dataset : $DATASET"
echo " samples : $SAMPLES"
echo " output  : $OUTPUT"
echo "=============================="

cd "$PROJECT_DIR"
python main.py \
    --config "$CONFIG" \
    --mode sensitivity \
    --sens_dataset "$DATASET" \
    --sens_samples "$SAMPLES" \
    --sens_output "$OUTPUT"

echo ""
echo "Sensitivity results saved to: $OUTPUT"
echo "Top 5 most sensitive layers:"
python - "$OUTPUT" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
ranked = data["ranked_by_sensitivity"]
for i, entry in enumerate(ranked[:5]):
    print(f"  #{i+1}  layer {entry['layer']:2d}  drop={entry['drop']:.4f}  acc={entry['accuracy']:.4f}")
EOF
