#!/bin/bash
#SBATCH --job-name=nn
#SBATCH -o nn.out
#SBATCH -e nn.err
source  /home/bingxing2/apps/package/pytorch/2.3.0+cu118_cp38/env.sh
module load anaconda
source activate generative
export PYTHONUNBUFFERED=1
python NN_EI_0.py
