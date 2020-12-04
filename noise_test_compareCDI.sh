#!/bin/bash

# read amples in .dat file
wfsensor=5
network="varnoise_10ewfstest_20x20wfs__3_4_ratio"

python CompareNN_MatlabBilinearInterp.py --network $network --net_type original --wfsensor $wfsensor --outfile

