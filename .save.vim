let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net2
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +68 CompareNN_MatlabBilinearInterp.py
badd +914 diffraction_functions.py
badd +4 matlab_cdi/seeded_run_CDI_noprocessing.m
badd +1 matlab_cdi/cdi_interpolation.m
badd +0 term://.//6541:/bin/bash
badd +18 term://.//6566:/bin/bash
badd +6 matlab_cdi/loaddata.m
badd +94 GetMeasuredDiffractionPattern.py
badd +30 TestPropagate.py
badd +643 diffraction_net.py
badd +36 matlab_cdi/seeded_reconst_func.m
badd +25 run_tests.sh
badd +23 zernike3/build/propagate.py
badd +16 zernike3/src/main.cpp
badd +1 zernike3/src/zernikedatagen.h
badd +183 term://.//34262:/bin/bash
badd +4 zernike3/build/viewdata.py
badd +3365 term://.//34908:/bin/bash
badd +0 term://.//35297:/bin/bash
badd +0 term://.//39975:/bin/bash
badd +59 zernike3/build/addnoise.py
badd +0 term://.//40578:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit matlab_cdi/seeded_run_CDI_noprocessing.m
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
if bufexists("term://.//6541:/bin/bash") | buffer term://.//6541:/bin/bash | else | edit term://.//6541:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 2864 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2864
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
let s:l = 33 - ((14 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
33
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m") | buffer ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | else | edit ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 47 - ((0 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
47
normal! 0
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py
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
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 39 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 39 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
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
let s:l = 112 - ((5 * winheight(0) + 3) / 6)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
112
normal! 0
lcd ~/diffraction_net2
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
12
normal! zo
22
normal! zo
53
normal! zo
93
normal! zo
96
normal! zo
99
normal! zo
let s:l = 97 - ((24 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
97
normal! 0
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
let s:l = 25 - ((18 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
25
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//40578:/bin/bash") | buffer term://.//40578:/bin/bash | else | edit term://.//40578:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 20 - ((19 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
20
normal! 040|
lcd ~/diffraction_net2
wincmd w
3wincmd w
exe '1resize ' . ((&lines * 6 + 24) / 49)
exe '2resize ' . ((&lines * 39 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 39 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 39 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/diffraction_net2/diffraction_net.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
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
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
argglobal
if bufexists("term://.//39975:/bin/bash") | buffer term://.//39975:/bin/bash | else | edit term://.//39975:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 918 - ((13 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
918
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
16
normal! zo
63
normal! zo
64
normal! zo
144
normal! zo
453
normal! zo
455
normal! zo
459
normal! zo
638
normal! zo
let s:l = 643 - ((4 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
643
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
63
normal! zo
64
normal! zo
144
normal! zo
453
normal! zo
455
normal! zo
459
normal! zo
let s:l = 525 - ((2 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
525
normal! 022|
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
63
normal! zo
64
normal! zo
144
normal! zo
201
normal! zo
453
normal! zo
455
normal! zo
459
normal! zo
709
normal! zo
714
normal! zo
let s:l = 145 - ((26 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
145
normal! 013|
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
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
if bufexists("term://.//34908:/bin/bash") | buffer term://.//34908:/bin/bash | else | edit term://.//34908:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 3600 - ((17 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
3600
normal! 040|
lcd ~/diffraction_net2
tabedit ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m
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
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
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
let s:l = 61 - ((14 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
61
normal! 04|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m") | buffer ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | else | edit ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 4 - ((3 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 0123|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m") | buffer ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | else | edit ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 56 - ((0 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
56
normal! 0
lcd ~/diffraction_net2
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
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
