#!/bin/bash
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=10G
#SBATCH --output=Job.%j.out
#SBATCH --error=Job.%j.err
#SBATCH --partition=aquila
#SBATCH --gres=gpu:2

module purge
module load anaconda3/4.0.0

python <  run.py
