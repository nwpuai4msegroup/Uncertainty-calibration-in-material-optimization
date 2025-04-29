#!/bin/bash
#SBATCH -J XGBOOST_EI_0
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -o %XGBOOST_EI_0.out
#SBATCH -e %XGBOOST_EI_0.err

date

module load anaconda/3
source activate base
python XGBOOST_EI_0.py 

date
