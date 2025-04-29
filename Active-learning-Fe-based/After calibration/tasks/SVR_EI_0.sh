#!/bin/bash
#SBATCH -J SVR_EI_0
#SBATCH -p CPU
#SBATCH -N 1
#SBATCH -n 15
#SBATCH -o %SVR_EI_0.out
#SBATCH -e %SVR_EI_0.err

date

module load anaconda/3
source activate base
python SVR_EI_0.py 

date
