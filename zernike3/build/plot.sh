#!/bin/bash
gnuplot -persist <<EOF
set term wxt
# set arrow nohead from 0,0 to 100,100 front lc "black"
plot 'opencvm1.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
# set arrow nohead from 0,0 to 100,100 front lc "black"
plot 'opencvm1_complex.dat' matrix with image
EOF

# gnuplot -persist <<EOF
# set term wxt
# plot 'sum_rows.dat' with line,\
# 	'sum_cols.dat' with line
# EOF
