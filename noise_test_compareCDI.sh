#!/bin/bash
pc=50
camera_noise="../../SquareWFtest/CameraNoise/1_1000/Bild_1.png"
cd zernike3/build/
python addnoise.py --infile test.hdf5 --outfile test_noise.hdf5 --peakcount $pc --cameraimage $camera_noise
cd ../..
python CompareNN_MatlabBilinearInterp.py
