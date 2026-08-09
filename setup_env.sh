#!/usr/bin/env bash
# setup_env.sh — one-shot environment bootstrap for sentinel/
#
# Run this ONCE after cloning:
#   bash setup_env.sh
#
# Requires: Python 3.11 or 3.12 (NOT 3.14 — box2d-py has no wheel for it yet)
# On macOS: install via https://www.python.org/downloads/release/python-3119/
#           or: brew install python@3.11  (after accepting Xcode license)
#
# Prerequisites on macOS:
#   sudo xcodebuild -license accept   ← must be done once in Terminal

set -e

PYTHON=${PYTHON:-python3.11}

echo "==> Checking Python version …"
$PYTHON --version

echo "==> Creating virtual environment (.venv) …"
$PYTHON -m venv .venv

echo "==> Activating …"
source .venv/bin/activate

echo "==> Upgrading pip …"
pip install --upgrade pip

echo "==> Installing swig (required to compile box2d-py) …"
pip install "swig==4.*"

echo "==> Installing project dependencies …"
pip install -r requirements.txt

echo ""
echo "Done! To activate the environment in your shell:"
echo "  source sentinel/.venv/bin/activate"
echo ""
echo "To generate the dataset:"
echo "  cd sentinel && python data_generation/generate_dataset.py --nominal 5 --fault 10"
