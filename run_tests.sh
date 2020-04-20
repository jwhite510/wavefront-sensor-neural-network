#!/bin/bash

# set -e

# generate dataset
cd zernike3/build/
rm ./*.hdf5
./a.out --name test.hdf5 --count 200
./a.out --name train.hdf5 --count 200
cd ../..

export batch_run_name=cdatanoNAN_200s_7
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

export batch_run_name=cdatanoNAN_200s_7
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

# generate dataset
cd zernike3/build/
rm ./*.hdf5
./a.out --name test.hdf5 --count 200
./a.out --name train.hdf5 --count 200
cd ../..

export batch_run_name=cdatanoNAN_200s_8
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

export batch_run_name=cdatanoNAN_200s_8
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network
