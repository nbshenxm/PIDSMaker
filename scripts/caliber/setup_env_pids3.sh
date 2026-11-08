#!/bin/bash
# Create the `pids3` conda env for PIDSMaker 3.x (torch 2.1.2 + CUDA 12.1).
# Mirrors the package pins in the upstream Dockerfile. Run on a LOGIN node
# (compute nodes have no internet). Leaves the old `pids` env untouched.
#
#   bash scripts/caliber/setup_env_pids3.sh 2>&1 | tee ~/pids3_setup.log
set -euo pipefail

ENV_NAME="${ENV_NAME:-pids3}"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "env ${ENV_NAME} already exists; installing/upgrading packages into it"
else
    conda create -y -n "${ENV_NAME}" -c conda-forge python=3.9 pip
fi
conda activate "${ENV_NAME}"

conda install -y -c conda-forge psycopg2 tqdm libstdcxx-ng libgcc-ng icu

pip install scikit-learn==1.6.1 networkx==3.2.1 xxhash==3.2.0 \
    graphviz==0.20.1 psutil scipy==1.13.1 matplotlib==3.9.4 \
    wandb==0.24.2 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 \
    cairocffi==1.7.0 wget==3.2

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

pip install torch_geometric==2.6.1 --no-cache-dir
# pyg_lib is omitted on purpose: its wheel needs glibc>=2.29 (cluster is RHEL8,
# glibc 2.28) and, when present, its failed load also disables torch_sparse.
pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
    torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir

pip install gensim==4.4.0 pytz==2024.1 pandas==2.3.3 yacs==0.1.8 transformers==4.39.3
pip install numpy==1.26.4
pip install gdown==5.2.0 umap-learn==0.5.6
pip install pytest==8.3.5 pyarrow

# NLTK data must be pre-fetched here: compute nodes cannot download it.
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || true

python -c "import torch, torch_geometric; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'pyg', torch_geometric.__version__)"
echo "pids3 env ready"
