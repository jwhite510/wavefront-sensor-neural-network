#!/bin/bash

# read amples in .dat file
wfsensor=0
network="wfstest_0"
pc=0
camera_noise="SquareWFtest/CameraNoise/1_1000/Bild_1.png"


noisecountv_arr=(50 100 150 0)
for cts in ${noisecountv_arr[@]}
do
	# generate a dataset with specific objects
	python datagen.py --samplesf datagen.dat --name specific_samples.hdf5 --wfsensor $wfsensor
	python addnoise.py --infile specific_samples.hdf5 --outfile specific_samples_noise.hdf5 --peakcount $pc --cameraimage $camera_noise --wfsensor $wfsensor
	python CompareNN_MatlabBilinearInterp.py --network $network --net_type original --pc $pc --wfsensor $wfsensor --outfile out_"$cts".p
done

python nice_plot_nn_iterative_compared_noise.py --wfsensor $wfsensor
