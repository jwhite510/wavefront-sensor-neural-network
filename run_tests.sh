#!/bin/bash

# set -e

source zernike3/loadmodules.sh

# generate dataset
cd zernike3/build/
rm ./*.hdf5
# ./a.out --name test.hdf5 --count 200
mpirun -np 2 a.out --count 200 --name test.hdf5 --buffersize 100
# ./a.out --name train.hdf5 --count 200
mpirun -np 20 a.out --count 20000 --name train.hdf5 --buffersize 100
cd ../..

export batch_run_name=mpidata_20000_3
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

export batch_run_name=mpidata_20000_3
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

export batch_run_name=mpidata_20000_3
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

export batch_run_name=mpidata_20000_3
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network
