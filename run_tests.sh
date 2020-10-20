#!/bin/bash

# set -e

# source zernike3/loadmodules.sh

declare -a runs=(
"visible_test_nr6" # name
"36000" # train samples
"50" # peak count
"original" # nr network


)
i=0
while (( $i < ${#runs[@]}))
do
	network=${runs[$i]}
	training_samples=${runs[$i+1]}
	pc=${runs[$i+2]}
	net_type=${runs[$i+3]}

	echo $network
	echo $training_samples

	# generate dataset
	rm ./${network}*.hdf5
	# create samples
	python datagen.py --count ${training_samples} --name ${network}_train.hdf5 --batch_size 100 --seed 345678 --wfsensor 0
	python datagen.py --count 200 --name ${network}_test.hdf5 --batch_size 100 --seed 8977 --wfsensor 0

	camera_noise="SquareWFtest/CameraNoise/1_1000/Bild_1.png"
	# add noise to samples
	python addnoise.py --infile ${network}_train.hdf5 --outfile ${network}_train_noise.hdf5 --peakcount $pc --cameraimage $camera_noise --wfsensor 0
	python addnoise.py --infile ${network}_test.hdf5 --outfile ${network}_test_noise.hdf5 --peakcount $pc --cameraimage $camera_noise --wfsensor 0

	echo ${network}
	python diffraction_net.py --name ${network} --net_type ${net_type} --wfsensor 0

	i=$i+4
done

