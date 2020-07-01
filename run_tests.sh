#!/bin/bash

# set -e

# source zernike3/loadmodules.sh

declare -a runs=(
"noise_test_1"
"36000"
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
	# create samples
	mpirun -np 2 a.out --count 200 --name test.hdf5 --buffersize 100 --seed 345678
	mpirun -np 20 a.out --count $training_samples --name train.hdf5 --buffersize 100 --seed 8977
	cd ../..

	peakcounts_arr=(10 20 30)
	camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
	for pc in "${peakcounts_arr[@]}"
	do
		# echo $pc
		cd zernike3/build/
		# add noise to samples
		python addnoise.py --infile train.hdf5 --outfile train_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
		python addnoise.py --infile test.hdf5 --outfile test_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
		cd ../..

		echo ${run_name}_peak-${pc}
		python diffraction_net.py ${run_name}_peak-${pc}
	done

	i=$i+2
done

