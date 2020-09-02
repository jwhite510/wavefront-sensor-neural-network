#!/bin/bash

rm -rf ./test_pc_*
rm -rf ./error_*.p

# for the infinity counts comparison case, use the network trained with 50 counts
camera_noise="SquareWFtest/CameraNoise/1_1000/Bild_1.png"
DIR="newdata_9_2"
network="_allwithlin_andscale_nrtest1_fixeccostf3"

rm -rf ./$DIR

python addnoise.py --infile ${network}_test.hdf5 --outfile ${network}_test_noise.hdf5 --peakcount 50 --cameraimage $camera_noise

python CompareNN_MatlabBilinearInterp.py --network $network --net_type nr --pc 0 --DIR $DIR

exit
