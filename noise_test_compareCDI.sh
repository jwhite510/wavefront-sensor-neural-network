#!/bin/bash


for k in $(seq 1 14)
do
	echo $k
	outfile="test_"$k".gif"
	rm ./tmp.py
	# write to the .dat file the parameters
	echo "import numpy as np" >> tmp.py
	echo "v=np.arange(-6,6+0.5,0.5).reshape(-1,1)" >> tmp.py
	# scale numbers:
	echo "s=np.array(np.shape(v)[0]*[0.5]).reshape(-1,1)" >> tmp.py
	let END=15-1 # total length 15, - scales and  -linspace
	for i in $(seq 1 $END)
	do
		if [ $i -eq $k ]; then # zernike term (1 or greater)
			echo "s=np.append(s,v,axis=1)" >> tmp.py
		else
			echo "s=np.append(s,np.zeros((np.shape(v)[0])).reshape(-1,1),axis=1)" >> tmp.py
		fi
	done
	# echo "s=np.append(s,np.zeros((np.shape(v)[0])).reshape(-1,1),axis=1)" >> tmp.py
	echo "np.savetxt('datagen.dat',s)" >> tmp.py
	python tmp.py

	# for the infinity counts comparison case, use the network trained with 50 counts
	camera_noise="SquareWFtest/CameraNoise/1_1000/Bild_1.png"
	DIR="data_9_30_linear"
	network="visible_test_nr2"
	pc=0

	rm -rf ./$DIR
	# make .dat file


	# generate a dataset with specific objects
	python datagen.py --samplesf datagen.dat --name specific_samples.hdf5
	python addnoise.py --infile specific_samples.hdf5 --outfile specific_samples_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
	python CompareNN_MatlabBilinearInterp.py --network $network --net_type nr --pc $pc --DIR $DIR --outfile $outfile
done

exit
