#!/bin/bash

# for the infinity counts comparison case, use the network trained with 50 counts
camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
cd zernike3/build/
python addnoise.py --infile test.hdf5 --outfile test_noise.hdf5 --peakcount 0 --cameraimage $camera_noise
cd ../..
python CompareNN_MatlabBilinearInterp.py --network noise_test_D_fixednorm_SQUARE6x6_VISIBLESETUP_NOCENTER_peak-50 --IMAGE_ANNOTATE _c_INF --SAVE_FOLDER test_7_20_cINF

pca=(50 10 5 2)
for pc in "${pca[@]}"
do
	cd zernike3/build/
	python addnoise.py --infile test.hdf5 --outfile test_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
	cd ../..
	python CompareNN_MatlabBilinearInterp.py --network noise_test_D_fixednorm_SQUARE6x6_VISIBLESETUP_NOCENTER_peak-$pc --IMAGE_ANNOTATE _c_$pc --SAVE_FOLDER test_7_20_c$pc
done
