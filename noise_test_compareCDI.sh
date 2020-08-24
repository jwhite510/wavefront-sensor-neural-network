#!/bin/bash

rm -rf ./test_pc_*
rm -rf ./error_*.p

# for the infinity counts comparison case, use the network trained with 50 counts
camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
DIR="8_13_20_test"
network="8_20_2020_center_intensity_N_1_3"

# remake test hdf5
cd zernike3/build/
rm ./${network}_test.hdf5
mpirun -np 2 a.out --count 200 --name ${network}_test.hdf5 --buffersize 100 --seed 345678
cd ../..

rm -rf ./$DIR
cd zernike3/build/
python addnoise.py --infile ${network}_test.hdf5 --outfile ${network}_test_noise.hdf5 --peakcount 50 --cameraimage $camera_noise
cd ../..
python CompareNN_MatlabBilinearInterp.py --network $network --pc 0 --DIR $DIR

exit
