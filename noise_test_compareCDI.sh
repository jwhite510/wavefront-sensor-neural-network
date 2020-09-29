#!/bin/bash

rm -rf ./test_pc_*
rm -rf ./error_*.p

# for the infinity counts comparison case, use the network trained with 50 counts
camera_noise="SquareWFtest/CameraNoise/1_1000/Bild_1.png"
DIR="data_9_29"
network="visible_test_nr2"

rm -rf ./$DIR

# generate a dataset with specific objects
python datagen.py --samplesf datagen.dat --name specific_samples.hdf5
python addnoise.py --infile specific_samples.hdf5 --outfile specific_samples_noise.hdf5 --peakcount 50 --cameraimage $camera_noise
python CompareNN_MatlabBilinearInterp.py --network $network --net_type nr --pc 0 --DIR $DIR

exit
