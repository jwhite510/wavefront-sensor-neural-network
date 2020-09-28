#!/bin/bash

# set -e

# source zernike3/loadmodules.sh

declare -a runs=(
"8_20_2020_xuvresult_13nmprop"
"36000"
)
i=0
while (( $i < ${#runs[@]}))
do
	network=${runs[$i]}
	training_samples=${runs[$i+1]}

	echo $network
	echo $training_samples

	# generate dataset
	cd zernike3/build/
	rm ./${network}*.hdf5
	# create samples
	mpirun -np 2 a.out --count 200 --name ${network}_test.hdf5 --buffersize 100
	mpirun -np 20 a.out --count $training_samples --name ${network}_train.hdf5 --buffersize 100
	cd ../..

	python diffraction_net.py ${network}

	i=$i+2
done

