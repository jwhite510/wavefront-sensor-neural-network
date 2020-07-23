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
badd +498 diffraction_net.py
badd +11 TestPropagate.py
badd +46 phase_normalization.py
badd +472 term://.//18637:/bin/bash
badd +67 CompareNN_MatlabBilinearInterp.py
badd +8 noise_test_compareCDI.sh
badd +81 term://.//11821:/bin/bash
badd +938 diffraction_functions.py
badd +288 matlab_cdi/seeded_run_CDI_noprocessing.m
badd +64 matlab_cdi/seeded_reconst_func.m
badd +46 term://.//11841:/bin/bash
badd +1 term://.//11933:/bin/bash
badd +34 ~/tf2test/main.py
badd +0 term://.//22570:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
if bufexists("term://.//11821:/bin/bash") | buffer term://.//11821:/bin/bash | else | edit term://.//11821:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 678 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
678
normal! 040|
lcd ~/diffraction_net2
tabedit ~/diffraction_net2/diffraction_net.py
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
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
argglobal
if bufexists("term://.//11933:/bin/bash") | buffer term://.//11933:/bin/bash | else | edit term://.//11933:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 248 - ((14 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
248
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
67
normal! zo
427
normal! zo
let s:l = 430 - ((7 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
430
normal! 0115|
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
19
normal! zo
67
normal! zo
252
normal! zo
305
normal! zo
427
normal! zo
547
normal! zo
549
normal! zo
553
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
575
normal! zo
577
normal! zo
579
normal! zo
589
normal! zo
594
normal! zo
631
normal! zo
727
normal! zo
734
normal! zo
let s:l = 457 - ((8 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
457
normal! 09|
lcd ~/diffraction_net2
wincmd w
3wincmd w
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net2/diffraction_net.py
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
setlocal fen
19
normal! zo
67
normal! zo
68
normal! zo
252
normal! zo
305
normal! zo
427
normal! zo
547
normal! zo
549
normal! zo
553
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
566
normal! zo
575
normal! zo
577
normal! zo
579
normal! zo
589
normal! zo
594
normal! zo
631
normal! zo
727
normal! zo
734
normal! zo
let s:l = 128 - ((22 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
128
normal! 09|
lcd ~/diffraction_net2
tabedit ~/tf2test/main.py
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
exe 'vert 1resize ' . ((&columns * 116 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 95 + 106) / 212)
argglobal
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
6
normal! zo
25
normal! zo
29
normal! zo
48
normal! zo
let s:l = 14 - ((9 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
14
normal! 09|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//22570:/bin/bash") | buffer term://.//22570:/bin/bash | else | edit term://.//22570:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 9240 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
9240
normal! 0
lcd ~/diffraction_net2
wincmd w
exe 'vert 1resize ' . ((&columns * 116 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 95 + 106) / 212)
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
