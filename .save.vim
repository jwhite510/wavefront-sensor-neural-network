let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +11 todo.txt
badd +5 todo_old.txt
badd +105 zernike3/src/main.cpp
badd +277 zernike3/src/pythonarrays.h
badd +414 zernike3/build/utility.py
badd +5 zernike3/runmpi.sh
badd +8 run_tests.sh
badd +100 zernike3/src/zernikedatagen.h
badd +1 diffraction_functions.py
badd +11 calibrate_measured_data.py
badd +451 diffraction_net.py
badd +84 zernike3/build/addnoise.py
badd +1 add_noise.sh
badd +46 zernike3/build/PropagateTF.py
badd +153 ~/.bashrc
badd +9973 term://.//7464:/bin/bash
badd +61 term://.//9163:/bin/bash
badd +77 CompareNN_MatlabBilinearInterp.py
badd +33 PropagateSphericalAperture.py
badd +18 zernike3/build/testprop.py
badd +96 GetMeasuredDiffractionPattern.py
badd +1 makeplots.sh
badd +31 TestPropagate.py
badd +20 zernike3/build/plot.py
badd +674 term://.//12063:/bin/bash
badd +485 term://.//27496:/bin/bash
badd +698 term://.//37257:/bin/bash
badd +25 zernike3/src/utility.h
badd +58 term://.//5915:/bin/bash
badd +6 term://.//33644:/bin/bash
badd +1 zernike3/build/propagate.py
badd +1 zernike3/build/viewdata.py
badd +1107 term://.//17764:/bin/bash
badd +10046 term://.//15795:/bin/bash
badd +3 zernike3/build/params_cu.dat
badd +1003 term://.//32480:/bin/bash
badd +8 term://.//44860:/bin/bash
badd +93 term://.//47191:/bin/bash
badd +1 5f64\ de79
badd +697 term://.//42303:/bin/bash
badd +370 term://.//35185:/bin/bash
badd +20 matlab_cdi/cdi_interpolation.m
badd +1242 term://.//35957:/bin/bash
badd +87 term://.//36567:/bin/bash
badd +581 term://.//40830:/bin/bash
badd +5 term://.//41019:/bin/bash
badd +2 noise_test_compareCDI.sh
badd +227 term://.//6625:/bin/bash
badd +244 term://.//6854:/bin/bash
badd +16 .gitignore
badd +109 term://.//7634:/bin/bash
badd +315 term://.//29299:/bin/bash
badd +59 term://.//29845:/bin/bash
badd +771 term://.//7422:/bin/bash
badd +39 multires_network.py
badd +756 ~/attosecond/v3/network3.py
badd +573 term://.//4687:/bin/bash
badd +53 term://.//698:/bin/bash
badd +10008 term://.//5839:/bin/bash
badd +265 term://.//41233:/bin/bash
badd +1 bof38
badd +1 bof3860a
badd +1 ae461dfb~205f21ffa13b2203ff233a2eb47da2443b747f719
badd +1 bof3860a~1
badd +209 term://.//1577:/bin/bash
badd +25 term://.//19633:/bin/bash
badd +46 term://.//44677:/bin/bash
badd +60 vortex.py
badd +143 zernike3/src/c_arrays.h
badd +12 term://.//6345:/bin/bash
badd +39 ~/bin/plotd.py
badd +53 term://.//10109:/bin/bash
badd +0 zernike3/build/plot2.py
badd +467 term://.//12413:/bin/bash
badd +3015 term://.//25578:/bin/bash
badd +0 term://.//46805:/bin/bash
badd +46 checkoutput.py
badd +2040 term://.//528:/bin/bash
badd +0 term://.//24730:/bin/bash
badd +420 term://.//3054:/bin/bash
badd +0 term://.//15616:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit checkoutput.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=2
setlocal fml=1
setlocal fdn=20
setlocal fen
6
normal! zo
14
normal! zo
16
normal! zo
26
normal! zo
let s:l = 35 - ((30 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
35
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//24730:/bin/bash") | buffer term://.//24730:/bin/bash | else | edit term://.//24730:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 264 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
264
normal! 039|
lcd ~/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/checkoutput.py
set splitbelow splitright
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 52 + 106) / 212)
exe '3resize ' . ((&lines * 8 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 52 + 106) / 212)
exe '4resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 68 + 106) / 212)
exe '6resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 77 + 106) / 212)
exe '7resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 7resize ' . ((&columns * 65 + 106) / 212)
exe '8resize ' . ((&lines * 5 + 24) / 49)
exe 'vert 8resize ' . ((&columns * 65 + 106) / 212)
exe '9resize ' . ((&lines * 6 + 24) / 49)
exe 'vert 9resize ' . ((&columns * 106 + 106) / 212)
exe '10resize ' . ((&lines * 6 + 24) / 49)
exe 'vert 10resize ' . ((&columns * 105 + 106) / 212)
argglobal
if bufexists("term://.//5839:/bin/bash") | buffer term://.//5839:/bin/bash | else | edit term://.//5839:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 10016 - ((15 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10016
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=2
setlocal fml=1
setlocal fdn=20
setlocal fen
26
normal! zo
let s:l = 36 - ((6 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
36
normal! 023|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//15616:/bin/bash") | buffer term://.//15616:/bin/bash | else | edit term://.//15616:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 189 - ((7 * winheight(0) + 4) / 8)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
189
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//528:/bin/bash") | buffer term://.//528:/bin/bash | else | edit term://.//528:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 2040 - ((6 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2040
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/run_tests.sh") | buffer ~/diffraction_net/run_tests.sh | else | edit ~/diffraction_net/run_tests.sh | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 8 - ((2 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
8
normal! 043|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/todo.txt") | buffer ~/diffraction_net/todo.txt | else | edit ~/diffraction_net/todo.txt | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 6 - ((5 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
6
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 113 - ((5 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
113
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_functions.py") | buffer ~/diffraction_net/diffraction_functions.py | else | edit ~/diffraction_net/diffraction_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
343
normal! zo
740
normal! zo
let s:l = 349 - ((2 * winheight(0) + 2) / 5)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
349
normal! 018|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//5915:/bin/bash") | buffer term://.//5915:/bin/bash | else | edit term://.//5915:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 58 - ((5 * winheight(0) + 3) / 6)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
58
normal! 054|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//46805:/bin/bash") | buffer term://.//46805:/bin/bash | else | edit term://.//46805:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 919 - ((5 * winheight(0) + 3) / 6)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
919
normal! 039|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 52 + 106) / 212)
exe '3resize ' . ((&lines * 8 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 52 + 106) / 212)
exe '4resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 68 + 106) / 212)
exe '6resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 77 + 106) / 212)
exe '7resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 7resize ' . ((&columns * 65 + 106) / 212)
exe '8resize ' . ((&lines * 5 + 24) / 49)
exe 'vert 8resize ' . ((&columns * 65 + 106) / 212)
exe '9resize ' . ((&lines * 6 + 24) / 49)
exe 'vert 9resize ' . ((&columns * 106 + 106) / 212)
exe '10resize ' . ((&lines * 6 + 24) / 49)
exe 'vert 10resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/diffraction_net.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal nofen
let s:l = 520 - ((18 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
520
normal! 021|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal nofen
let s:l = 457 - ((15 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
457
normal! 0127|
lcd ~/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/diffraction_net.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal nofen
let s:l = 445 - ((22 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
445
normal! 0
lcd ~/diffraction_net
tabedit ~/diffraction_net/diffraction_net.py
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 19 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 19 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 26 + 24) / 49)
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
let s:l = 510 - ((8 * winheight(0) + 9) / 19)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
510
normal! 038|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
let s:l = 71 - ((6 * winheight(0) + 9) / 19)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
71
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
let s:l = 444 - ((9 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
444
normal! 011|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 19 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 19 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 26 + 24) / 49)
tabedit ~/diffraction_net/diffraction_functions.py
set splitbelow splitright
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 53 + 106) / 212)
exe '4resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 105 + 106) / 212)
exe '6resize ' . ((&lines * 14 + 24) / 49)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
343
normal! zo
let s:l = 505 - ((8 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
505
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 115 - ((6 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
115
normal! 0116|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 100 - ((2 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
100
normal! 0117|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 77 - ((3 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
77
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 483 - ((10 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
483
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 119 - ((5 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
119
normal! 09|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 53 + 106) / 212)
exe '4resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 105 + 106) / 212)
exe '6resize ' . ((&lines * 14 + 24) / 49)
tabedit ~/diffraction_net/zernike3/src/main.cpp
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
38
normal! zo
103
normal! zo
let s:l = 105 - ((13 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
105
normal! 019|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
501
normal! zo
653
normal! zo
660
normal! zo
679
normal! zo
let s:l = 693 - ((21 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
693
normal! 011|
lcd ~/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/zernike3/src/zernikedatagen.h
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 571 - ((9 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
571
normal! 026|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
731
normal! zo
731
normal! zo
732
normal! zo
let s:l = 725 - ((27 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
725
normal! 03|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
336
normal! zo
363
normal! zo
let s:l = 369 - ((8 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
369
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
108
normal! zo
156
normal! zo
157
normal! zo
171
normal! zo
173
normal! zo
336
normal! zo
344
normal! zo
731
normal! zo
731
normal! zo
let s:l = 354 - ((9 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
354
normal! 0
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/zernike3/src/c_arrays.h
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 5 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 32 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 52 + 106) / 212)
exe '4resize ' . ((&lines * 19 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 52 + 106) / 212)
exe '6resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 106 + 106) / 212)
exe '7resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 7resize ' . ((&columns * 105 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
28
normal! zo
148
normal! zo
151
normal! zo
160
normal! zo
164
normal! zo
let s:l = 146 - ((1 * winheight(0) + 2) / 5)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
146
normal! 026|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
16
normal! zo
42
normal! zo
48
normal! zo
52
normal! zo
65
normal! zo
69
normal! zo
70
normal! zo
83
normal! zo
84
normal! zo
108
normal! zo
156
normal! zo
157
normal! zo
161
normal! zo
171
normal! zo
173
normal! zo
731
normal! zo
731
normal! zo
732
normal! zo
let s:l = 717 - ((9 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
717
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
16
normal! zo
42
normal! zo
48
normal! zo
52
normal! zo
65
normal! zo
69
normal! zo
70
normal! zo
83
normal! zo
84
normal! zo
108
normal! zo
156
normal! zo
157
normal! zo
161
normal! zo
171
normal! zo
173
normal! zo
501
normal! zo
526
normal! zo
541
normal! zo
588
normal! zo
597
normal! zo
603
normal! zo
616
normal! zo
653
normal! zo
660
normal! zo
670
normal! zo
679
normal! zo
698
normal! zo
731
normal! zo
731
normal! zo
let s:l = 525 - ((20 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
525
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/build/plot2.py") | buffer ~/diffraction_net/zernike3/build/plot2.py | else | edit ~/diffraction_net/zernike3/build/plot2.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
6
normal! zo
let s:l = 10 - ((9 * winheight(0) + 9) / 19)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//10109:/bin/bash") | buffer term://.//10109:/bin/bash | else | edit term://.//10109:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 132 - ((17 * winheight(0) + 9) / 18)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
132
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//6345:/bin/bash") | buffer term://.//6345:/bin/bash | else | edit term://.//6345:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 378 - ((6 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
378
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/runmpi.sh") | buffer ~/diffraction_net/zernike3/runmpi.sh | else | edit ~/diffraction_net/zernike3/runmpi.sh | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 5 - ((4 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
5
normal! 0
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 5 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 32 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 52 + 106) / 212)
exe '4resize ' . ((&lines * 19 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 52 + 106) / 212)
exe '6resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 106 + 106) / 212)
exe '7resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 7resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/vortex.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
41
normal! zo
66
normal! zo
let s:l = 58 - ((19 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
58
normal! 05|
lcd ~/diffraction_net
tabnext 3
set stal=1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOFc
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
