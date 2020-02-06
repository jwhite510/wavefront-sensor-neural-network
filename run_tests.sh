#!/bin/bash

set -e

# generate dataset
rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/dataset_generators/with_phase_subtraction.py
export batch_run_name=AUWNKL_withphasesubtraction1
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=AUWNKL_withphasesubtraction2
sbatch --wait submit_gpu_job.slurm # start training network

rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/dataset_generators/withOUT_phase_subtraction.py
export batch_run_name=AUWNKL_withOUTphasesubtraction1
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=AUWNKL_withOUTphasesubtraction2
sbatch --wait submit_gpu_job.slurm # start training network

rm ./*.hdf5
echo generating dataset
~/python_compiled/bin/python3 ~/projects/diffraction_run/dataset_generators/with_phase_subtraction_increaseradius.py
export batch_run_name=AUWNKL_withphasesubtraction_increaseradius1
sbatch --wait submit_gpu_job.slurm # start training network
export batch_run_name=AUWNKL_withphasesubtraction_increaseradius2
sbatch --wait submit_gpu_job.slurm # start training network

