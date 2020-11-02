#!/bin/bash

# read amples in .dat file
wfsensor=2
network="varnoise_10ewfstest_2"

python CompareNN_MatlabBilinearInterp.py --network $network --net_type original --wfsensor $wfsensor --outfile

