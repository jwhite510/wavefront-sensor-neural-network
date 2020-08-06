#!/bin/bash

rm -rf ./test_pc_*
rm -rf ./error_*.p

# for the infinity counts comparison case, use the network trained with 50 counts
camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
DIR="8_6_20_test"
rm -rf ./$DIR
cd zernike3/build/
python addnoise.py --infile test.hdf5 --outfile test_noise.hdf5 --peakcount 50 --cameraimage $camera_noise
cd ../..
python CompareNN_MatlabBilinearInterp.py --network vis1_2_peak-50 --pc 0 --DIR $DIR

exit

pca=(50 10 5 2)
for pc in "${pca[@]}"
do
	cd zernike3/build/
	python addnoise.py --infile test.hdf5 --outfile test_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
	cd ../..
	python CompareNN_MatlabBilinearInterp.py --network noise_test_D_fixednorm_SQUARE6x6_VISIBLESETUP_NOCENTER_peak-$pc --pc $pc
done
python plot_error_compare.py --pc ${pca[@]}
