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
# set term png

while 1{
	do for [i=1:9]{
		set output sprintf('im%d.png',i)
		# plot 'opencvm1_complex_after1.dat' matrix with image
		plot sprintf('opencvm1_complex_after%d.dat',i) matrix with image
		# set title sprintf('%d', i)
		pause 0.2
		reread
	}
}
EOF
# gnuplot -persist <<EOF
# set term wxt
# plot 'sum_rows.dat' with line,\
# 	'sum_cols.dat' with line
# EOF
