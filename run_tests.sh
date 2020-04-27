#!/bin/bash

# set -e

source zernike3/loadmodules.sh

declare -a runs=(
"mpidata_26000_20_lr0001"
"26000"

"mpidata_28000_21_lr0001"
"29000"

"mpidata_36000_22_lr0001"
"35000"

"mpidata_36000_23_lr0001"
"35000"

)
i=0
while (( $i < ${#runs[@]}))
do
	run_name=${runs[$i]}
	training_samples=${runs[$i+1]}

	echo $run_name
	echo $training_samples

	# generate dataset
	cd zernike3/build/
	rm ./*.hdf5
	# ./a.out --name test.hdf5 --count 200
	mpirun -np 2 a.out --count 200 --name test.hdf5 --buffersize 100
	# ./a.out --name train.hdf5 --count 200
	mpirun -np 20 a.out --count $training_samples --name train.hdf5 --buffersize 100
	cd ../..

	export batch_run_name=$run_name
	echo submtting network training job $batch_run_name
	sbatch --wait submit_gpu_job.slurm # start training network
	export batch_run_name=$run_name
	echo submtting network training job $batch_run_name
	sbatch --wait submit_gpu_job.slurm # start training network
	export batch_run_name=$run_name
	echo submtting network training job $batch_run_name
	sbatch --wait submit_gpu_job.slurm # start training network
	export batch_run_name=$run_name
	echo submtting network training job $batch_run_name
	sbatch --wait submit_gpu_job.slurm # start training network


	i=$i+2
done



# # generate dataset
# cd zernike3/build/
# rm ./*.hdf5
# # ./a.out --name test.hdf5 --count 200
# mpirun -np 2 a.out --count 200 --name test.hdf5 --buffersize 100
# # ./a.out --name train.hdf5 --count 200
# mpirun -np 20 a.out --count 20000 --name train.hdf5 --buffersize 100
# cd ../..
# 
# export batch_run_name=mpidata_20000_15_lr00005
# echo submtting network training job $batch_run_name
# sbatch --wait submit_gpu_job.slurm # start training network
# export batch_run_name=mpidata_20000_15_lr00005
# echo submtting network training job $batch_run_name
# sbatch --wait submit_gpu_job.slurm # start training network
# export batch_run_name=mpidata_20000_15_lr00005
# echo submtting network training job $batch_run_name
# sbatch --wait submit_gpu_job.slurm # start training network
# export batch_run_name=mpidata_20000_15_lr00005
# echo submtting network training job $batch_run_name
# sbatch --wait submit_gpu_job.slurm # start training network


