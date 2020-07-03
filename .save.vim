let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +6 todo.txt
badd +5 todo_old.txt
badd +89 zernike3/src/main.cpp
badd +277 zernike3/src/pythonarrays.h
badd +412 zernike3/build/utility.py
badd +5 zernike3/runmpi.sh
badd +35 run_tests.sh
badd +553 zernike3/src/zernikedatagen.h
badd +36 diffraction_functions.py
badd +11 calibrate_measured_data.py
badd +143 diffraction_net.py
badd +1 .git/index
badd +26 zernike3/build/addnoise.py
badd +3 add_noise.sh
badd +46 zernike3/build/PropagateTF.py
badd +153 ~/.bashrc
badd +783 ~/.vimrc
badd +0 term://.//7464:/bin/bash
badd +61 term://.//9163:/bin/bash
badd +114 CompareNN_MatlabBilinearInterp.py
badd +33 PropagateSphericalAperture.py
badd +18 zernike3/build/testprop.py
badd +96 GetMeasuredDiffractionPattern.py
badd +32 TestPropagate.py
badd +1 zernike3/build/plot.py
badd +674 term://.//12063:/bin/bash
badd +485 term://.//27496:/bin/bash
badd +698 term://.//37257:/bin/bash
badd +18 zernike3/src/utility.h
badd +0 term://.//43436:/bin/bash
badd +6 term://.//33644:/bin/bash
badd +1 zernike3/build/propagate.py
badd +1 zernike3/build/viewdata.py
badd +1107 term://.//17764:/bin/bash
badd +10046 term://.//15795:/bin/bash
badd +3 zernike3/build/params_cu.dat
badd +1003 term://.//32480:/bin/bash
badd +464 term://.//44922:/bin/bash
badd +93 term://.//47191:/bin/bash
badd +0 fugitive:///home/jonathon/diffraction_net/.git//cf6641b84fc1f7ecd942101823a7a26545b5e11c/zernike3/src/zernikedatagen.h
badd +100 fugitive:///home/jonathon/diffraction_net/.git//5f64a569c833a26c0a52b85fbaa584f316f7f24d/zernike3/src/zernikedatagen.h
badd +0 5f64\ de79
badd +0 term://.//32485:/bin/bash
badd +697 term://.//42303:/bin/bash
badd +0 term://.//42802:/bin/bash
badd +370 term://.//35185:/bin/bash
badd +20 matlab_cdi/cdi_interpolation.m
badd +1 fugitive:///home/jonathon/diffraction_net/.git//0/CompareNN_MatlabBilinearInterp.py
badd +1242 term://.//35957:/bin/bash
badd +0 term://.//36285:/bin/bash
badd +87 term://.//36567:/bin/bash
badd +581 term://.//40830:/bin/bash
badd +0 term://.//41019:/bin/bash
badd +0 noise_test_compareCDI.sh
badd +0 term://.//46463:/bin/bash
badd +227 term://.//6625:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit run_tests.sh
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
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
exe '1resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 116 + 106) / 212)
exe '2resize ' . ((&lines * 28 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 68 + 106) / 212)
exe '3resize ' . ((&lines * 28 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 47 + 106) / 212)
exe '4resize ' . ((&lines * 42 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 95 + 106) / 212)
exe '5resize ' . ((&lines * 3 + 24) / 49)
argglobal
if bufexists("term://.//7464:/bin/bash") | buffer term://.//7464:/bin/bash | else | edit term://.//7464:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 10013 - ((12 * winheight(0) + 6) / 13)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10013
normal! 039|
lcd ~/diffraction_net
wincmd w
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 26 - ((12 * winheight(0) + 14) / 28)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
26
normal! 09|
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
let s:l = 2 - ((1 * winheight(0) + 14) / 28)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2
normal! 04|
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
328
normal! zo
let s:l = 342 - ((16 * winheight(0) + 21) / 42)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
342
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//43436:/bin/bash") | buffer term://.//43436:/bin/bash | else | edit term://.//43436:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 136 - ((2 * winheight(0) + 1) / 3)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
136
normal! 054|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 116 + 106) / 212)
exe '2resize ' . ((&lines * 28 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 68 + 106) / 212)
exe '3resize ' . ((&lines * 28 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 47 + 106) / 212)
exe '4resize ' . ((&lines * 42 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 95 + 106) / 212)
exe '5resize ' . ((&lines * 3 + 24) / 49)
tabedit ~/diffraction_net/CompareNN_MatlabBilinearInterp.py
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
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
exe '1resize ' . ((&lines * 5 + 24) / 49)
exe '2resize ' . ((&lines * 40 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 141 + 106) / 212)
exe '3resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
argglobal
if bufexists("term://.//36285:/bin/bash") | buffer term://.//36285:/bin/bash | else | edit term://.//36285:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1845 - ((2 * winheight(0) + 2) / 5)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1845
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
54
normal! zo
100
normal! zo
let s:l = 67 - ((15 * winheight(0) + 20) / 40)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
67
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/noise_test_compareCDI.sh") | buffer ~/diffraction_net/noise_test_compareCDI.sh | else | edit ~/diffraction_net/noise_test_compareCDI.sh | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 2 - ((1 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//46463:/bin/bash") | buffer term://.//46463:/bin/bash | else | edit term://.//46463:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 309 - ((6 * winheight(0) + 6) / 13)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
309
normal! 0
lcd ~/diffraction_net
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 5 + 24) / 49)
exe '2resize ' . ((&lines * 40 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 141 + 106) / 212)
exe '3resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/diffraction_net/.git/index
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
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
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe '2resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
argglobal
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=1
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 13 - ((12 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
13
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("fugitive:///home/jonathon/diffraction_net/.git//0/CompareNN_MatlabBilinearInterp.py") | buffer fugitive:///home/jonathon/diffraction_net/.git//0/CompareNN_MatlabBilinearInterp.py | else | edit fugitive:///home/jonathon/diffraction_net/.git//0/CompareNN_MatlabBilinearInterp.py | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 62 - ((11 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
62
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/CompareNN_MatlabBilinearInterp.py") | buffer ~/diffraction_net/CompareNN_MatlabBilinearInterp.py | else | edit ~/diffraction_net/CompareNN_MatlabBilinearInterp.py | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 67 - ((15 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
67
normal! 05|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe '2resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/diffraction_functions.py
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
29
normal! zo
906
normal! zo
let s:l = 904 - ((0 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
904
normal! 0
lcd ~/diffraction_net
tabedit ~/diffraction_net/zernike3/build/addnoise.py
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
setlocal fdl=2
setlocal fml=1
setlocal fdn=20
setlocal fen
12
normal! zo
26
normal! zo
45
normal! zo
47
normal! zo
52
normal! zo
53
normal! zo
let s:l = 47 - ((22 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
47
normal! 013|
lcd ~/diffraction_net
tabnew
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
if bufexists("term://.//41019:/bin/bash") | buffer term://.//41019:/bin/bash | else | edit term://.//41019:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 5 - ((4 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
5
normal! 026|
lcd ~/diffraction_net
tabedit ~/diffraction_net/zernike3/build/addnoise.py
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
setlocal fdl=2
setlocal fml=1
setlocal fdn=20
setlocal fen
12
normal! zo
26
normal! zo
45
normal! zo
47
normal! zo
52
normal! zo
53
normal! zo
let s:l = 75 - ((36 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
75
normal! 017|
lcd ~/diffraction_net
tabedit ~/diffraction_net/zernike3/src/zernikedatagen.h
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 10 + 24) / 49)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
275
normal! zo
284
normal! zo
501
normal! zo
541
normal! zo
565
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
565
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
let s:l = 101 - ((17 * winheight(0) + 17) / 35)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
101
normal! 015|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("fugitive:///home/jonathon/diffraction_net/.git//cf6641b84fc1f7ecd942101823a7a26545b5e11c/zernike3/src/zernikedatagen.h") | buffer fugitive:///home/jonathon/diffraction_net/.git//cf6641b84fc1f7ecd942101823a7a26545b5e11c/zernike3/src/zernikedatagen.h | else | edit fugitive:///home/jonathon/diffraction_net/.git//cf6641b84fc1f7ecd942101823a7a26545b5e11c/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 100 - ((9 * winheight(0) + 8) / 17)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
100
normal! 03|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("fugitive:///home/jonathon/diffraction_net/.git//5f64a569c833a26c0a52b85fbaa584f316f7f24d/zernike3/src/zernikedatagen.h") | buffer fugitive:///home/jonathon/diffraction_net/.git//5f64a569c833a26c0a52b85fbaa584f316f7f24d/zernike3/src/zernikedatagen.h | else | edit fugitive:///home/jonathon/diffraction_net/.git//5f64a569c833a26c0a52b85fbaa584f316f7f24d/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 99 - ((5 * winheight(0) + 8) / 17)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
99
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
enew
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 10 + 24) / 49)
tabedit ~/diffraction_net/zernike3/src/main.cpp
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
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
exe '1resize ' . ((&lines * 8 + 24) / 49)
exe '2resize ' . ((&lines * 37 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 60 + 106) / 212)
exe '3resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe '5resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 80 + 106) / 212)
exe '6resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 80 + 106) / 212)
argglobal
if bufexists("term://.//44922:/bin/bash") | buffer term://.//44922:/bin/bash | else | edit term://.//44922:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 788 - ((3 * winheight(0) + 4) / 8)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
788
normal! 0
lcd ~/diffraction_net
wincmd w
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
71
normal! zo
89
normal! zo
103
normal! zo
109
normal! zo
116
normal! zo
120
normal! zo
let s:l = 88 - ((10 * winheight(0) + 18) / 37)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
88
normal! 011|
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
let s:l = 100 - ((7 * winheight(0) + 9) / 18)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
100
normal! 03|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/build/utility.py") | buffer ~/diffraction_net/zernike3/build/utility.py | else | edit ~/diffraction_net/zernike3/build/utility.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
401
normal! zo
414
normal! zo
423
normal! zo
let s:l = 411 - ((2 * winheight(0) + 9) / 18)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
411
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//42802:/bin/bash") | buffer term://.//42802:/bin/bash | else | edit term://.//42802:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 34 - ((8 * winheight(0) + 9) / 18)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
34
normal! 010|
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
541
normal! zo
565
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
565
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
let s:l = 562 - ((11 * winheight(0) + 9) / 18)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
562
normal! 05|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 8 + 24) / 49)
exe '2resize ' . ((&lines * 37 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 60 + 106) / 212)
exe '3resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe '5resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 80 + 106) / 212)
exe '6resize ' . ((&lines * 18 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 80 + 106) / 212)
tabedit ~/diffraction_net/zernike3/src/zernikedatagen.h
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 158 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 52 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 52 + 106) / 212)
exe '4resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 53 + 106) / 212)
exe '6resize ' . ((&lines * 10 + 24) / 49)
argglobal
if bufexists("term://.//32485:/bin/bash") | buffer term://.//32485:/bin/bash | else | edit term://.//32485:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1088 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1088
normal! 039|
lcd ~/diffraction_net
wincmd w
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
35,35fold
34,35fold
24,38fold
44,44fold
49,49fold
48,49fold
66,66fold
65,66fold
52,66fold
73,74fold
77,78fold
70,79fold
69,80fold
85,85fold
84,85fold
83,86fold
42,86fold
90,93fold
16,94fold
98,102fold
120,120fold
119,120fold
127,127fold
126,127fold
145,145fold
144,145fold
130,145fold
150,152fold
149,153fold
158,158fold
157,158fold
156,159fold
162,162fold
161,163fold
108,166fold
174,176fold
173,176fold
180,182fold
179,182fold
171,183fold
196,201fold
207,208fold
206,209fold
214,214fold
213,215fold
205,215fold
221,222fold
220,223fold
228,228fold
227,229fold
219,229fold
233,233fold
188,234fold
251,260fold
264,264fold
264,264fold
268,268fold
240,269fold
276,276fold
286,291fold
284,292fold
303,303fold
275,303fold
311,312fold
315,315fold
309,316fold
308,317fold
330,331fold
329,332fold
322,332fold
359,359fold
358,359fold
344,359fold
365,366fold
364,367fold
363,371fold
375,378fold
336,379fold
395,409fold
416,418fold
416,418fold
455,455fold
455,455fold
454,455fold
453,456fold
414,456fold
460,463fold
383,464fold
473,483fold
488,489fold
469,490fold
495,496fold
526,528fold
531,539fold
526,539fold
589,589fold
588,589fold
604,604fold
603,604fold
597,605fold
541,610fold
644,644fold
616,648fold
656,656fold
664,664fold
671,671fold
670,671fold
660,671fold
679,681fold
679,681fold
693,693fold
699,699fold
698,699fold
703,703fold
706,706fold
717,717fold
719,719fold
653,720fold
726,726fold
501,727fold
501
normal! zo
541
normal! zo
let s:l = 554 - ((9 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
554
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
501
normal! zo
526
normal! zo
let s:l = 104 - ((8 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
104
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/TestPropagate.py") | buffer ~/diffraction_net/TestPropagate.py | else | edit ~/diffraction_net/TestPropagate.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
10,13fold
24,24fold
33,34fold
33,34fold
50,54fold
50,54fold
61,65fold
61,65fold
75,79fold
75,79fold
60,82fold
90,90fold
99,99fold
17,102fold
17
normal! zo
let s:l = 31 - ((11 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
31
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
enew
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
lcd ~/diffraction_net
wincmd w
argglobal
enew
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 158 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 52 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 52 + 106) / 212)
exe '4resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 53 + 106) / 212)
exe '6resize ' . ((&lines * 10 + 24) / 49)
tabnext 2
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
