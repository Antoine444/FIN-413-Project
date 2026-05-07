#!/bin/bash

# Create cache and output directories 
mkdir -p cache output figures

# Create and activate a new conda environment
conda create -n uniswap_project python=3.11 -y
conda init uniswap_project
conda activate uniswap_project

# Install dependencies
pip install web3 pandas pyarrow python-dotenv requests tqdm
