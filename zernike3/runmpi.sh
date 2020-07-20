#!/bin/bash

set -e
cd build
mpirun -np 1 a.out --count 300 --name train.hdf5 --buffersize 20 --seed 1234567
