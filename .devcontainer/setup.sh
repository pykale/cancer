#!/usr/bin/env bash
# Prepare a Codespace or dev container for kalecancer.
set -euo pipefail

echo "Installing kalecancer (CPU PyTorch)..."
pip install --upgrade pip
# Codespaces have no GPU; the CPU wheel is much smaller and installs faster.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,survival]"

echo
echo "Verifying..."
python -c "import kalecancer, torch, kale, torchsurv; print('kalecancer', kalecancer.__version__, '| torch', torch.__version__)"
pytest -q

cat <<'EOF'

Ready.

  Fetch a small cohort from the public HANCOCK archives:
    kalecancer data pull --patients 50

  Run the pipeline on it:
    kalecancer wsi-survival --source hancock --patients 50 --preset quick

This container has no GPU. Use it for development, tests and small demonstration
runs; train full cohorts on a GPU machine.
EOF
