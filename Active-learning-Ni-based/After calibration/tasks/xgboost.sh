#!/bin/bash
#SBATCH --job-name=xgboost
#SBATCH -o xgboost.out
#SBATCH -e xgboost.err
source  /home/bingxing2/apps/package/pytorch/2.3.0+cu118_cp38/env.sh
module load anaconda
source activate generative
export PYTHONUNBUFFERED=1
python XGBOOST_EI_0.py
