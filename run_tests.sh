#!/bin/bash

# set -e

# generate dataset
rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/generate_data_withphasesubtract.py
export batch_run_name=LIUJH_scalar_phase_withphasesubtract1
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=LIUJH_scalar_phase_withphasesubtract2
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network

rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/generate_data_withoutphasesubtract.py
export batch_run_name=LIUJH_scalar_phase_withoutphasesubtract1
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=LIUJH_scalar_phase_withoutphasesubtract2
echo submtting network training job $batch_run_name
sbatch --wait submit_gpu_job.slurm # start training network


