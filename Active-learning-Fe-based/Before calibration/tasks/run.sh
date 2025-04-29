#!/bin/bash
#SBATCH --job-name=rf
#SBATCH -o rf.out
#SBATCH -e rf.err
source  /home/bingxing2/apps/package/pytorch/2.3.0+cu118_cp38/env.sh
module load anaconda
source activate generative
export PYTHONUNBUFFERED=1
python RF_EI_0.py
