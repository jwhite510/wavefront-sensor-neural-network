let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net2
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +39 vortex.py
badd +127 diffraction_net.py
badd +11 TestPropagate.py
badd +11 phase_normalization.py
badd +472 term://.//18637:/bin/bash
badd +277 CompareNN_MatlabBilinearInterp.py
badd +11 noise_test_compareCDI.sh
badd +2718 term://.//11821:/bin/bash
badd +134 diffraction_functions.py
badd +288 matlab_cdi/seeded_run_CDI_noprocessing.m
badd +64 matlab_cdi/seeded_reconst_func.m
badd +46 term://.//11841:/bin/bash
badd +14 ~/tf2test/main.py
badd +0 deconv_test.py
badd +577 term://.//2641:/bin/bash
badd +351 term://.//29283:/bin/bash
badd +256 term://.//16208:/bin/bash
badd +8 plot_error_compare.py
badd +3211 term://.//20698:/bin/bash
badd +197 term://.//29039:/bin/bash
badd +0 term://.//41853:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit CompareNN_MatlabBilinearInterp.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 9 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
53
normal! zo
59
normal! zo
244
normal! zo
let s:l = 274 - ((6 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
274
normal! 05|
lcd ~/diffraction_net2
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
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/plot_error_compare.py") | buffer ~/diffraction_net2/plot_error_compare.py | else | edit ~/diffraction_net2/plot_error_compare.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
7
normal! zo
13
normal! zo
14
normal! zo
22
normal! zo
37
normal! zo
let s:l = 11 - ((1 * winheight(0) + 4) / 9)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 05|
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
let s:l = 14 - ((12 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
14
normal! 09|
lcd ~/diffraction_net2
wincmd w
4wincmd w
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 9 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
tabedit ~/diffraction_net2/plot_error_compare.py
set splitbelow splitright
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
exe 'vert 1resize ' . ((&columns * 136 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 75 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 75 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
7
normal! zo
13
normal! zo
14
normal! zo
22
normal! zo
37
normal! zo
let s:l = 30 - ((24 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
30
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//41853:/bin/bash") | buffer term://.//41853:/bin/bash | else | edit term://.//41853:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 101 - ((22 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
101
normal! 040|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//20698:/bin/bash") | buffer term://.//20698:/bin/bash | else | edit term://.//20698:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 3211 - ((21 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
3211
normal! 040|
lcd ~/diffraction_net2
wincmd w
exe 'vert 1resize ' . ((&columns * 136 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 75 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 75 + 106) / 212)
tabedit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py
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
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
53
normal! zo
59
normal! zo
244
normal! zo
let s:l = 58 - ((0 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
58
normal! 09|
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
53
normal! zo
59
normal! zo
244
normal! zo
let s:l = 122 - ((15 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
122
normal! 09|
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
53
normal! zo
59
normal! zo
138
normal! zo
190
normal! zo
244
normal! zo
let s:l = 137 - ((0 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
137
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
53
normal! zo
59
normal! zo
138
normal! zo
190
normal! zo
244
normal! zo
let s:l = 184 - ((11 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
184
normal! 05|
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
tabedit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py
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
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
53
normal! zo
59
normal! zo
138
normal! zo
190
normal! zo
244
normal! zo
let s:l = 152 - ((15 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
152
normal! 05|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/diffraction_functions.py") | buffer ~/diffraction_net2/diffraction_functions.py | else | edit ~/diffraction_net2/diffraction_functions.py | endif
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
let s:l = 86 - ((25 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
86
normal! 0
lcd ~/diffraction_net2
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
tabnext 1
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
