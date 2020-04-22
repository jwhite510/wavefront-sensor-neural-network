#!/bin/bash

set -e
cd build
mpirun -np 5 a.out --count 300 --name train.hdf5 --buffersize 20
