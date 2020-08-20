#!/bin/bash

# set -e

# source zernike3/loadmodules.sh

declare -a runs=(
"8_20_2020_center_intensity" # name
"36000" # train samples
"50" # peak count
)
i=0
while (( $i < ${#runs[@]}))
do
	network=${runs[$i]}
	training_samples=${runs[$i+1]}
	pc=${runs[$i+2]}

	echo $network
	echo $training_samples

	# generate dataset
	cd zernike3/build/
	rm ./${network}*.hdf5
	# create samples
	mpirun -np 2 a.out --count 200 --name ${network}_test.hdf5 --buffersize 100 --seed 345678
	mpirun -np 20 a.out --count $training_samples --name ${network}_train.hdf5 --buffersize 100 --seed 8977
	cd ../..

	camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
	# echo $pc
	cd zernike3/build/
	# add noise to samples
	python addnoise.py --infile ${network}_train.hdf5 --outfile ${network}_train_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
	python addnoise.py --infile ${network}_test.hdf5 --outfile ${network}_test_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
	cd ../..

	echo ${network}
	python diffraction_net.py ${network}

	i=$i+3
done

