#!/bin/bash
#SBATCH -J NN_EI_0
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -o %NN_EI_0.out
#SBATCH -e %NN_EI_0.err

date

module load anaconda/3
source activate base
python NN_EI_0.py 

date
