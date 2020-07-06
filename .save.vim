let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net2
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +85 CompareNN_MatlabBilinearInterp.py
badd +150 diffraction_functions.py
badd +61 matlab_cdi/seeded_run_CDI_noprocessing.m
badd +1 matlab_cdi/cdi_interpolation.m
badd +18 term://.//6566:/bin/bash
badd +232 run_network.py
badd +1 matlab_cdi/loaddata.m
badd +94 GetMeasuredDiffractionPattern.py
badd +30 TestPropagate.py
badd +792 diffraction_net.py
badd +36 matlab_cdi/seeded_reconst_func.m
badd +33 run_tests.sh
badd +23 zernike3/build/propagate.py
badd +124 zernike3/src/main.cpp
badd +1 zernike3/src/zernikedatagen.h
badd +183 term://.//34262:/bin/bash
badd +4 zernike3/build/viewdata.py
badd +3600 term://.//34908:/bin/bash
badd +2044 term://.//35297:/bin/bash
badd +918 term://.//39975:/bin/bash
badd +64 zernike3/build/addnoise.py
badd +430 zernike3/build/utility.py
badd +10017 term://.//8161:/bin/bash
badd +0 term://.//10802:/bin/bash
badd +15 term://.//44183:/bin/bash
badd +10 noise_test_compareCDI.sh
badd +152 term://.//46048:/bin/bash
badd +96 term://.//3021:/bin/bash
badd +0 summary.txt
badd +956 term://.//16607:/bin/bash
badd +0 term://.//28844:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit run_tests.sh
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
exe 'vert 2resize ' . ((&columns * 146 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 65 + 106) / 212)
argglobal
if bufexists("term://.//10802:/bin/bash") | buffer term://.//10802:/bin/bash | else | edit term://.//10802:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 179 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
179
normal! 0
lcd ~/diffraction_net2
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
let s:l = 23 - ((12 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
23
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//8161:/bin/bash") | buffer term://.//8161:/bin/bash | else | edit term://.//8161:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 10016 - ((33 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10016
normal! 040|
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 146 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 65 + 106) / 212)
tabedit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py
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
exe '1resize ' . ((&lines * 6 + 24) / 49)
exe '2resize ' . ((&lines * 39 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 103 + 106) / 212)
exe '3resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 108 + 106) / 212)
exe '4resize ' . ((&lines * 27 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 108 + 106) / 212)
argglobal
if bufexists("term://.//35297:/bin/bash") | buffer term://.//35297:/bin/bash | else | edit term://.//35297:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 3276 - ((5 * winheight(0) + 3) / 6)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
3276
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
13,13fold
23,26fold
22,26fold
33,36fold
32,36fold
43,46fold
42,46fold
12,48fold
65,68fold
98,98fold
98,98fold
101,101fold
101,101fold
104,104fold
104,104fold
53,126fold
12
normal! zo
22
normal! zo
53
normal! zo
98
normal! zo
101
normal! zo
104
normal! zo
let s:l = 65 - ((13 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
65
normal! 05|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//46048:/bin/bash") | buffer term://.//46048:/bin/bash | else | edit term://.//46048:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 309 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
309
normal! 047|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/noise_test_compareCDI.sh") | buffer ~/diffraction_net2/noise_test_compareCDI.sh | else | edit ~/diffraction_net2/noise_test_compareCDI.sh | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 15 - ((14 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
15
normal! 09|
lcd ~/diffraction_net2
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 6 + 24) / 49)
exe '2resize ' . ((&lines * 39 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 103 + 106) / 212)
exe '3resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 108 + 106) / 212)
exe '4resize ' . ((&lines * 27 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 108 + 106) / 212)
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
if bufexists("term://.//28844:/bin/bash") | buffer term://.//28844:/bin/bash | else | edit term://.//28844:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 19 - ((18 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
19
normal! 040|
lcd ~/diffraction_net2
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
if bufexists("term://.//16607:/bin/bash") | buffer term://.//16607:/bin/bash | else | edit term://.//16607:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 915 - ((3 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
915
normal! 026|
lcd ~/diffraction_net2
tabedit ~/diffraction_net2/noise_test_compareCDI.sh
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
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
let s:l = 14 - ((13 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
14
normal! 09|
lcd ~/diffraction_net2
tabedit ~/diffraction_net2/zernike3/build/addnoise.py
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
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
26
normal! zo
45
normal! zo
47
normal! zo
52
normal! zo
let s:l = 47 - ((5 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
47
normal! 013|
lcd ~/diffraction_net2
tabedit ~/diffraction_net2/summary.txt
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
if bufexists("term://.//3021:/bin/bash") | buffer term://.//3021:/bin/bash | else | edit term://.//3021:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 96 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
96
normal! 040|
lcd ~/diffraction_net2
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
let s:l = 5 - ((4 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
5
normal! 09|
lcd ~/diffraction_net2
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
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
