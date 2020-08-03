let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net2
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 vortex.py
badd +1 diffraction_net.py
badd +11 TestPropagate.py
badd +11 phase_normalization.py
badd +472 term://.//18637:/bin/bash
badd +408 CompareNN_MatlabBilinearInterp.py
badd +1 noise_test_compareCDI.sh
badd +2718 term://.//11821:/bin/bash
badd +195 diffraction_functions.py
badd +1 matlab_cdi/seeded_run_CDI_noprocessing.m
badd +64 matlab_cdi/seeded_reconst_func.m
badd +46 term://.//11841:/bin/bash
badd +14 ~/tf2test/main.py
badd +1 deconv_test.py
badd +577 term://.//2641:/bin/bash
badd +351 term://.//29283:/bin/bash
badd +256 term://.//16208:/bin/bash
badd +52 plot_error_compare.py
badd +197 term://.//29039:/bin/bash
badd +257 term://.//10947:/bin/bash
badd +9580 term://.//7899:/bin/bash
badd +1063 term://.//33683:/bin/bash
badd +433 zernike3/build/utility.py
badd +55 GetMeasuredDiffractionPattern.py
badd +1531 term://.//5219:/bin/bash
badd +332 term://.//23820:/bin/bash
badd +1313 term://.//29861:/bin/bash
badd +2918 term://.//37071:/bin/bash
badd +314 term://.//26133:/bin/bash
badd +1 HEAD
badd +183 term://.//36638:/bin/bash
badd +1 HEAD\ vortex_beam
badd +1007 term://.//36802:/bin/bash
badd +1 .save.vim
badd +1 multires_network.py
badd +1 run_tests.sh
badd +92 zernike3/build/addnoise.py
badd +1 zernike3/build/plot2.py
badd +3 zernike3/runmpi.sh
badd +138 zernike3/src/c_arrays.h
badd +49 zernike3/src/main.cpp
badd +673 zernike3/src/zernikedatagen.h
badd +1 term://.//7961:/bin/bash
badd +528 term://.//46136:/bin/bash
badd +46 ~/.vimrc
badd +539 term://.//5689:/bin/bash
badd +50 ~/pythontf_cpu_venv/req.txt
badd +402 term://.//8027:/bin/bash
badd +230 term://.//3972:/bin/bash
badd +1170 term://.//4796:/bin/bash
badd +106 _main.py
badd +3928 term://.//8115:/bin/bash
badd +1 live_capture/TIS.py
badd +1 live_capture/live_stream_test.py
badd +260 ~/.vim/plugged/vim-fugitive/doc/fugitive.txt
badd +15 __Tagbar__.226
badd +1 .git/index
badd +0 fugitive:///home/jonathon/diffraction_net2/.git//0/_main.py
argglobal
%argdel
$argadd ./
set stal=2
edit diffraction_net.py
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
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
exe '5resize ' . ((&lines * 7 + 24) / 49)
argglobal
if bufexists("term://.//7899:/bin/bash") | buffer term://.//7899:/bin/bash | else | edit term://.//7899:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 23 - ((22 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
23
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 55 - ((7 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
55
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/diffraction_net.py") | buffer ~/diffraction_net2/diffraction_net.py | else | edit ~/diffraction_net2/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
16
normal! zo
64
normal! zo
65
normal! zo
237
normal! zo
454
normal! zo
456
normal! zo
712
normal! zo
717
normal! zo
755
normal! zo
let s:l = 134 - ((7 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
134
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/run_tests.sh") | buffer ~/diffraction_net2/run_tests.sh | else | edit ~/diffraction_net2/run_tests.sh | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 8 - ((2 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
8
normal! 07|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//7961:/bin/bash") | buffer term://.//7961:/bin/bash | else | edit term://.//7961:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 7 - ((6 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
7
normal! 0
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
exe '5resize ' . ((&lines * 7 + 24) / 49)
tabedit ~/diffraction_net2/noise_test_compareCDI.sh
set splitbelow splitright
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 5resize ' . ((&columns * 70 + 106) / 212)
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
let s:l = 11 - ((4 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 08|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/CompareNN_MatlabBilinearInterp.py") | buffer ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | else | edit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
54
normal! zo
136
normal! zo
166
normal! zo
172
normal! zo
174
normal! zo
176
normal! zo
182
normal! zo
let s:l = 147 - ((9 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
147
normal! 013|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//8027:/bin/bash") | buffer term://.//8027:/bin/bash | else | edit term://.//8027:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 11 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/CompareNN_MatlabBilinearInterp.py") | buffer ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | else | edit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
15
normal! zo
54
normal! zo
166
normal! zo
172
normal! zo
174
normal! zo
176
normal! zo
182
normal! zo
364
normal! zo
374
normal! zo
385
normal! zo
413
normal! zo
420
normal! zo
let s:l = 416 - ((12 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
416
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/CompareNN_MatlabBilinearInterp.py") | buffer ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | else | edit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
54
normal! zo
60
normal! zo
136
normal! zo
166
normal! zo
172
normal! zo
174
normal! zo
176
normal! zo
182
normal! zo
let s:l = 170 - ((5 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
170
normal! 017|
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 5resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/diffraction_net2/_main.py
set splitbelow splitright
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
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 83 - ((12 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
83
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//8115:/bin/bash") | buffer term://.//8115:/bin/bash | else | edit term://.//8115:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 22 - ((21 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
22
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/_main.py") | buffer ~/diffraction_net2/_main.py | else | edit ~/diffraction_net2/_main.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 182 - ((18 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
182
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/_main.py") | buffer ~/diffraction_net2/_main.py | else | edit ~/diffraction_net2/_main.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
26
normal! zo
236
normal! zo
let s:l = 244 - ((15 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
244
normal! 09|
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/diffraction_net2/_main.py
set splitbelow splitright
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
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 12 - ((11 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
12
normal! 028|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/_main.py") | buffer ~/diffraction_net2/_main.py | else | edit ~/diffraction_net2/_main.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
15
normal! zo
26
normal! zo
27
normal! zo
let s:l = 106 - ((8 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
106
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/live_capture/TIS.py") | buffer ~/diffraction_net2/live_capture/TIS.py | else | edit ~/diffraction_net2/live_capture/TIS.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
18
normal! zo
let s:l = 19 - ((18 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
19
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/live_capture/live_stream_test.py") | buffer ~/diffraction_net2/live_capture/live_stream_test.py | else | edit ~/diffraction_net2/live_capture/live_stream_test.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
9
normal! zo
28
normal! zo
28
normal! zo
29
normal! zo
29
normal! zo
let s:l = 36 - ((18 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
36
normal! 025|
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/diffraction_net2/_main.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 85 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 85 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 40 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
15
normal! zo
26
normal! zo
27
normal! zo
let s:l = 101 - ((14 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
101
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/_main.py") | buffer ~/diffraction_net2/_main.py | else | edit ~/diffraction_net2/_main.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
15
normal! zo
26
normal! zo
27
normal! zo
let s:l = 75 - ((16 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
75
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
enew
file ~/diffraction_net2/__Tagbar__.4
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
lcd ~/diffraction_net2
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 85 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 85 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 40 + 106) / 212)
tabedit ~/diffraction_net2/.git/index
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
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
let s:l = 16 - ((5 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
16
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("fugitive:///home/jonathon/diffraction_net2/.git//0/_main.py") | buffer fugitive:///home/jonathon/diffraction_net2/.git//0/_main.py | else | edit fugitive:///home/jonathon/diffraction_net2/.git//0/_main.py | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 234 - ((128 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
234
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/_main.py") | buffer ~/diffraction_net2/_main.py | else | edit ~/diffraction_net2/_main.py | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 245 - ((130 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
245
normal! 0
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabnext 5
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
