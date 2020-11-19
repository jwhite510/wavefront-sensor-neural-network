#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=slurm-%j.out
#
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

##SBATCH --partition=gpu_p100
##SBATCH --partition=gpu_v100
#SBATCH --partition=gpu_test
# env

# # ~/python_compiled/bin/python3 --version
~/python_compiled/bin/python3 diffraction_net.py $batch_run_name


