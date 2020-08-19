#!/bin/bash

# set -e

# source zernike3/loadmodules.sh

declare -a runs=(
# "newgputest"
# "200"
"xuvresult_1"
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
	# ./a.out --name test.hdf5 --count 200
	mpirun -np 2 a.out --count 200 --name test.hdf5 --buffersize 100
	# ./a.out --name train.hdf5 --count 200
	mpirun -np 20 a.out --count $training_samples --name train.hdf5 --buffersize 100
	cd ../..

	python diffraction_net.py $run_name

	i=$i+2
done

