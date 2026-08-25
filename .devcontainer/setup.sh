#!/usr/bin/env bash
# Prepare a Codespace or dev container for kalecancer.
set -euo pipefail

echo "Installing kalecancer (CPU PyTorch)..."
pip install --upgrade pip
# Codespaces have no GPU; the CPU wheel is much smaller and installs faster.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

echo
echo "Verifying..."
python -c "import kalecancer, torch, kale; print('kalecancer', kalecancer.__version__, '| torch', torch.__version__)"
pytest -q

cat <<'EOF'

Ready.

  Fetch a small cohort from the public HANCOCK archives and run on it:
    python examples/wsi_survival/main.py \
        --cfg examples/wsi_survival/configs/hancock_primary_tumour_quick.yaml \
        DATASET.SOURCE hancock DATASET.PATIENTS 50

  Run on data already on disk:
    kalecancer wsi-survival --features <dir> --clinical <file> --preset quick

This container has no GPU. Use it for development, tests and small demonstration
runs; train full cohorts on a GPU machine.
EOF
