#!/bin/bash
set -euo pipefail

ENV_NAME="echorna"

if [ $# -gt 0 ]; then
	case "$1" in
		--envname)
			if [ $# -ne 2 ]; then echo "Usage: $0 [--envname <envname>]"; exit 1; fi
			ENV_NAME="$2"
			;;
		*) echo "Unknown argument: $1"; exit 1 ;;
	esac
fi

# --- Locate conda -------------------------------------------------------------
posixify() { if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s' "$1"; fi; }

CONDA_SH=""
# 1) conda already on PATH -> derive its base directory.
if command -v conda >/dev/null 2>&1; then
	base="$(conda info --base 2>/dev/null || true)"
	base="$(posixify "$base")"
	[ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ] && CONDA_SH="$base/etc/profile.d/conda.sh"
fi
# 2) Otherwise search common install locations.
if [ -z "$CONDA_SH" ]; then
	candidates=(
		"$HOME/miniconda3" "$HOME/anaconda3" "$HOME/Miniconda3" "$HOME/Anaconda3"
		"$HOME/miniforge3" "$HOME/mambaforge"
		"/c/ProgramData/miniconda3" "/c/ProgramData/anaconda3"
		"/c/miniconda3" "/c/anaconda3"
		"/opt/conda" "/opt/miniconda3" "/opt/anaconda3"
		"/usr/local/miniconda3" "/usr/local/anaconda3"
	)
	# Add Windows %USERPROFILE% / %LOCALAPPDATA% based locations when present.
	for w in "${USERPROFILE:-}" "${LOCALAPPDATA:-}"; do
		[ -n "$w" ] && candidates+=("$(posixify "$w")/miniconda3" "$(posixify "$w")/anaconda3")
	done
	for c in "${candidates[@]}"; do
		if [ -f "$c/etc/profile.d/conda.sh" ]; then
			CONDA_SH="$c/etc/profile.d/conda.sh"
			break
		fi
	done
fi
if [ -z "$CONDA_SH" ]; then
	echo "Error: conda not found. Install Miniconda/Anaconda, or run 'conda init' and restart your shell." >&2
	exit 1
fi

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
set -u
echo "Using conda: $CONDA_SH"

# --- Create / reuse the environment ------------------------------------------
CONDA_BASE="$(dirname "$(dirname "$(dirname "$CONDA_SH")")")"
if [ -d "$CONDA_BASE/envs/$ENV_NAME" ] || [ -d "${HOME:-}/.conda/envs/$ENV_NAME" ]; then
	echo "Conda environment '$ENV_NAME' already exists; reusing it."
else
	echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
	conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "Activating environment..."
set +u
conda activate "$ENV_NAME"
set -u

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing PyTorch 2.4.1 with CUDA 11.8..."
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing PyTorch Geometric packages..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

echo "Installing Python packages..."
pip install "numpy<2" "pandas<2.2" "fair-esm==2.0.0" "biotite==1.2.0" rich rna-fm pyyaml huggingface-hub "biopython==1.79"

# Patch a deprecated biotite call in fair-esm's inverse_folding/util.py
echo "Patching deprecated function in fair-esm..."
python - <<'PYEOF'
import importlib.util, os, sys, re
spec = importlib.util.find_spec("esm")
if spec is None or spec.origin is None:
    sys.stderr.write("Could not locate esm package.\n"); sys.exit(1)
path = os.path.join(os.path.dirname(spec.origin), "inverse_folding", "util.py")
if not os.path.isfile(path):
    sys.stderr.write("Could not locate util.py at %s\n" % path); sys.exit(1)
with open(path, "r", encoding="utf-8") as f:
    text = f.read()
if re.search(r"\bfilter_backbone\b", text):
    text = re.sub(r"\bfilter_backbone\b", "filter_peptide_backbone", text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Patched: %s" % path)
else:
    print("No 'filter_backbone' found in %s (already patched or upstream changed); skipping." % path)
PYEOF

echo "Verifying installation..."
python -c "import torch, torch_scatter, torch_sparse, torch_cluster, torch_geometric, fm, yaml, numpy; import esm, esm.inverse_folding.util"

set +u
conda deactivate
set -u
echo "Installation complete."
echo "Please activate the environment before use: conda activate $ENV_NAME"