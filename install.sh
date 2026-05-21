#!/bin/bash
set -e

INSTALL_DIR="$HOME/.local/bin"
ENV_NAME="echorna"

for arg in "$@"; do
	case "$arg" in
		--install-dir=*) INSTALL_DIR="${arg#*=}" ;;
		--env-name=*)    ENV_NAME="${arg#*=}" ;;
		*) echo "Unknown argument: $arg"; exit 1 ;;
	esac
done

mkdir -p "$INSTALL_DIR"

echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
conda create -n "$ENV_NAME" python=3.10 -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Upgrading pip..."
python -m ensurepip --upgrade

echo "Installing PyTorch 2.4.0 with CUDA 11.8..."
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing PyTorch Geometric packages..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

echo "Installing Python packages..."
pip install numpy "pandas<2.2" "fair-esm==2.0.0" "biotite==1.2.0" rich rna-fm pyyaml

echo "Installing Biopython..."
conda install -y biopython=1.78

# replace deprecated function
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
sed -i '12s/filter_backbone/filter_peptide_backbone/' "$SITE_PACKAGES/esm/inverse_folding/util.py"

echo "Verifying installation..."
python -c "import torchvision; import torch_scatter; import fm; import esm"

conda deactivate

echo "Installation complete."
echo "Please activate the environment before use: conda activate $ENV_NAME"