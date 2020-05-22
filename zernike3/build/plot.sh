#!/bin/bash
# gnuplot -persist <<EOF
# set term wxt
# # set arrow nohead from 0,0 to 100,100 front lc "black"
# plot 'opencvm1.dat' matrix with image
# EOF

gnuplot -persist <<EOF
set term wxt
# set arrow nohead from 0,0 to 100,100 front lc "black"
plot 'opencvm1_complex_before.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after1.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after1.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after2.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after3.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after4.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after5.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after6.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after7.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after8.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after9.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'opencvm1_complex_after10.dat' matrix with image
EOF

# gnuplot -persist <<EOF
# set term wxt
# plot 'sum_rows.dat' with line,\
# 	'sum_cols.dat' with line
# EOF
