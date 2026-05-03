#!/usr/bin/env bash
# One-time environment setup on the Linux remote machine
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up Python virtual environment …"
cd "$PROJECT_DIR"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete.  Activate the venv with:"
echo "  source ${PROJECT_DIR}/venv/bin/activate"
echo ""
echo "To authenticate with HuggingFace Hub (required for LLaMA-2):"
echo "  huggingface-cli login"
