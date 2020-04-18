#!/bin/bash

# set -e

# generate dataset
# rm ./*.hdf5
# echo generating dataset
# ~/python_compiled/bin/python3 generate_data.py

export batch_run_name=cdatanoNAN3
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

export batch_run_name=cdatanoNAN3
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network



