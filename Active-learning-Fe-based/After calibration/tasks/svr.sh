#!/bin/bash
#SBATCH --job-name=svr
#SBATCH -o svr.out
#SBATCH -e svr.err
source  /home/bingxing2/apps/package/pytorch/2.3.0+cu118_cp38/env.sh
module load anaconda
source activate generative
export PYTHONUNBUFFERED=1
python SVR_EI_0.py
