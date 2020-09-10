#!/bin/bash

rm -rf ./test_pc_*
rm -rf ./error_*.p

# for the infinity counts comparison case, use the network trained with 50 counts
camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
DIR="8_25_20_test_xuv"
network="8_20_2020_xuvresult"
rm -rf ./$DIR
# cd zernike3/build/
# python addnoise.py --infile ${network}_test.hdf5 --outfile ${network}_test_noise.hdf5 --peakcount 50 --cameraimage $camera_noise
# cd ../..

cd zernike3/build/
make
rm ./${network}*.hdf5
# create samples
mpirun -np 2 a.out --count 200 --name ${network}_test.hdf5 --buffersize 100
mpirun -np 2 a.out --count 200 --name ${network}_train.hdf5 --buffersize 100
cd ../..

python CompareNN_MatlabBilinearInterp.py --network $network --pc 0 --DIR $DIR

exit
