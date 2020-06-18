#!/bin/bash

cd zernike3/build/
python addnoise.py --infile train.hdf5 --outfile train_noise.hdf5 --peakcount 10 --cameraimage ../../SquareWFtest/CameraNoise/1_1000/Bild_1.png
