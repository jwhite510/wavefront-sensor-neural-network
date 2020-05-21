#!/bin/bash
gnuplot -persist <<EOF
set term wxt
plot 'opencvm1.dat' matrix with image
EOF

gnuplot -persist <<EOF
set term wxt
plot 'sum_rows.dat' with line,\
	'sum_cols.dat' with line
EOF
