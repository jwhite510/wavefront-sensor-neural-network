#!/bin/bash
gnuplot -persist <<EOF
set term wxt
plot 'Adif_in.dat' matrix with image
EOF
gnuplot -persist <<EOF
set term wxt
plot 'Adif_in_scaled.dat' matrix with image
EOF
gnuplot -persist <<EOF
set term wxt
plot 'Adif_in_scaled_rot.dat' matrix with image
EOF
