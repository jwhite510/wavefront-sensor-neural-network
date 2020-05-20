set term x11 0
plot 'Adif_in.dat' matrix with image

set term x11 1
plot 'Adif_in_scaled.dat' matrix with image

pause 0.1
reread

