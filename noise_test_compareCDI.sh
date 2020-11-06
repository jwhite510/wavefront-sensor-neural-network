#!/bin/bash

# read amples in .dat file
wfsensor=0
network="xuvdata2_varnoise_wfstest_0"
pc=0
camera_noise="SquareWFtest/CameraNoise/1_1000/Bild_1.png"

# generate a dataset with specific objects
# python CompareNN_MatlabBilinearInterp.py --network $network --net_type original --pc $pc --wfsensor $wfsensor --outfile out_"$cts".p

pca=(50 10 5 2)
for pc in "${pca[@]}"
do
	python addnoise.py --infile $network"_test.hdf5" --outfile $network"_test_"$pc"_.hdf5" --peakcount $pc --cameraimage $camera_noise --wfs $wfsensor
	python CompareNN_MatlabBilinearInterp.py --network $network --pc $pc --wfs $wfsensor --net_type original
done
python plot_error_compare.py --pc ${pca[@]} --wfs $wfsensor

