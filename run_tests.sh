#!/bin/bash

# set -e

# generate dataset
rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/with_phase_subtraction.py
export batch_run_name=AUWNKL_withphasesubtraction1
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=AUWNKL_withphasesubtraction2
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/withOUT_phase_subtraction.py
export batch_run_name=AUWNKL_withOUTphasesubtraction1
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=AUWNKL_withOUTphasesubtraction2
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/with_phase_subtraction_increaseradius.py
export batch_run_name=AUWNKL_withphasesubtraction_increaseradius1
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=AUWNKL_withphasesubtraction_increaseradius2
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

