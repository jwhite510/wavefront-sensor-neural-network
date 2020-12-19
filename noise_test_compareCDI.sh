#!/bin/bash

# read amples in .dat file
wfsensor=2
network="256_6x6_square_wfs_test"

python CompareNN_MatlabBilinearInterp.py --network $network --net_type original --wfsensor $wfsensor --outfile

