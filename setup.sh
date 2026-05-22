#!/bin/bash
set -e

# Create output directories
mkdir -p cache output figures

# Create and activate a new conda environment
conda create -n uniswap_project python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uniswap_project

# Install dependencies
pip install web3 pandas pyarrow python-dotenv requests tqdm matplotlib
