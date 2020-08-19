#!/bin/bash

rm -rf ./test_pc_*
rm -rf ./error_*.p

# for the infinity counts comparison case, use the network trained with 50 counts
camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
DIR="8_19_20_HDRtest"
rm -rf ./$DIR
cd zernike3/build/
python addnoise.py --infile test.hdf5 --outfile test_noise.hdf5 --peakcount 50 --cameraimage $camera_noise
cd ../..
python CompareNN_MatlabBilinearInterp.py --network net3_test_center_intensity_peak-50 --pc 0 --DIR $DIR

exit
