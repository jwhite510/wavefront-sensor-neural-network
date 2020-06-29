let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net2
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 term://.//6483:/bin/bash
badd +80 CompareNN_MatlabBilinearInterp.py
badd +923 diffraction_functions.py
badd +240 matlab_cdi/seeded_run_CDI_noprocessing.m
badd +1 term://.//6426:/bin/bash
badd +44 matlab_cdi/cdi_interpolation.m
badd +0 term://.//6541:/bin/bash
badd +18 term://.//6566:/bin/bash
badd +0 .git/index
badd +0 fugitive:///home/jonathon/diffraction_net2/.git//0/matlab_cdi/seeded_run_CDI_noprocessing.m
argglobal
%argdel
$argadd ./
set stal=2
edit diffraction_functions.py
set splitbelow splitright
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
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 141 + 106) / 212)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
argglobal
if bufexists("term://.//6426:/bin/bash") | buffer term://.//6426:/bin/bash | else | edit term://.//6426:/bin/bash | endif
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
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
899
normal! zo
906
normal! zo
919
normal! zo
let s:l = 926 - ((24 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
926
normal! 05|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/CompareNN_MatlabBilinearInterp.py") | buffer ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | else | edit ~/diffraction_net2/CompareNN_MatlabBilinearInterp.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
10
normal! zo
20
normal! zo
51
normal! zo
let s:l = 64 - ((15 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
64
normal! 05|
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("term://.//6483:/bin/bash") | buffer term://.//6483:/bin/bash | else | edit term://.//6483:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 35 - ((9 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
35
normal! 0
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 141 + 106) / 212)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m
set splitbelow splitright
wincmd _ | wincmd |
split
wincmd _ | wincmd |
split
2wincmd k
wincmd w
wincmd _ | wincmd |
vsplit
1wincmd h
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
exe '5resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 105 + 106) / 212)
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
let s:l = 967 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
967
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
let s:l = 71 - ((6 * winheight(0) + 8) / 17)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
71
normal! 05|
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
let s:l = 51 - ((10 * winheight(0) + 8) / 17)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
51
normal! 05|
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
let s:l = 224 - ((13 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
224
normal! 09|
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
let s:l = 229 - ((5 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
229
normal! 09|
lcd ~/diffraction_net2
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
exe '5resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 105 + 106) / 212)
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
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 30 + 24) / 49)
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
let s:l = 9 - ((8 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
9
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("fugitive:///home/jonathon/diffraction_net2/.git//0/matlab_cdi/seeded_run_CDI_noprocessing.m") | buffer fugitive:///home/jonathon/diffraction_net2/.git//0/matlab_cdi/seeded_run_CDI_noprocessing.m | else | edit fugitive:///home/jonathon/diffraction_net2/.git//0/matlab_cdi/seeded_run_CDI_noprocessing.m | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 225 - ((14 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
225
normal! 0
lcd ~/diffraction_net2
wincmd w
argglobal
if bufexists("~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m") | buffer ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | else | edit ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 242 - ((15 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
242
normal! 0
lcd ~/diffraction_net2
wincmd w
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net2/matlab_cdi/seeded_run_CDI_noprocessing.m
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
let s:l = 245 - ((37 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
245
normal! 09|
lcd ~/diffraction_net2
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
