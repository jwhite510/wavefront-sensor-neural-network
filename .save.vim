let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Projects/diffraction_net/zernike3
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +3 src/main.cpp
badd +4 CMakeLists.txt
badd +1 ~/Projects/diffraction_net/run_tests.sh
badd +57 src/testprop.cpp
badd +46 term://.//10858:/bin/bash
badd +626 src/zernikedatagen.h
badd +1 run.sh
badd +457 /usr/include/fftw3.h
badd +11 build/testprop.py
badd +19 build/utility.py
badd +46 term://.//9648:/bin/bash
badd +46 term://.//9640:/bin/bash
badd +6 src/python.txt
badd +1103 term://.//17293:/bin/bash
badd +190 build/diffraction_functions.py
badd +5 build/convert_svg_to_png.py
badd +1 ~/Projects/diffraction_net/.gitignore
badd +1 term://.//27130:/bin/bash
badd +1 ~/Projects/diffraction_net/run.sh
badd +1 term://.//11136:/bin/bash
badd +224 term://.//11016:/bin/bash
badd +6 ~/Projects/diffraction_net/testpropagation.py
badd +46 term://.//11188:/bin/bash
badd +89 src/pythonarrays.h
badd +93 src/c_arrays.h
badd +4 ~/.vimrc
badd +283 term://.//28056:/bin/bash
badd +2 ~/.editrc
badd +2 ~/Projects/diffraction_net/debug.gdb
badd +1 src
badd +1 ~/Projects/diffraction_net/.save.vim
badd +161 term://.//15771:/bin/bash
badd +171 term://.//16116:/bin/bash
argglobal
%argdel
$argadd build/
set stal=2
edit src/zernikedatagen.h
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
exe '1resize ' . ((&lines * 5 + 24) / 49)
exe '2resize ' . ((&lines * 40 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 12 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 53 + 106) / 212)
exe '4resize ' . ((&lines * 12 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 27 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 106 + 106) / 212)
argglobal
if bufexists("term://.//11016:/bin/bash") | buffer term://.//11016:/bin/bash | else | edit term://.//11016:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 201 - ((2 * winheight(0) + 2) / 5)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
201
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
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
let s:l = 289 - ((20 * winheight(0) + 20) / 40)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
289
normal! 07|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/build/testprop.py") | buffer ~/Projects/diffraction_net/zernike3/build/testprop.py | else | edit ~/Projects/diffraction_net/zernike3/build/testprop.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 51 - ((5 * winheight(0) + 6) / 12)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
51
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/build/testprop.py") | buffer ~/Projects/diffraction_net/zernike3/build/testprop.py | else | edit ~/Projects/diffraction_net/zernike3/build/testprop.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 27 - ((4 * winheight(0) + 6) / 12)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
27
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/testprop.cpp") | buffer ~/Projects/diffraction_net/zernike3/src/testprop.cpp | else | edit ~/Projects/diffraction_net/zernike3/src/testprop.cpp | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
let s:l = 53 - ((5 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
53
normal! 03|
lcd ~/Projects/diffraction_net
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 5 + 24) / 49)
exe '2resize ' . ((&lines * 40 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 12 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 53 + 106) / 212)
exe '4resize ' . ((&lines * 12 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 27 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 106 + 106) / 212)
tabedit ~/Projects/diffraction_net/zernike3/build/testprop.py
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
exe '2resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe '5resize ' . ((&lines * 7 + 24) / 49)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=1
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 13 - ((0 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
13
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/build/testprop.py") | buffer ~/Projects/diffraction_net/zernike3/build/testprop.py | else | edit ~/Projects/diffraction_net/zernike3/build/testprop.py | endif
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
14
normal! zo
23
normal! zo
28
normal! zo
52
normal! zo
55
normal! zo
56
normal! zo
63
normal! zo
64
normal! zo
65
normal! zo
66
normal! zo
69
normal! zo
73
normal! zo
87
normal! zo
105
normal! zo
let s:l = 92 - ((6 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
92
normal! 013|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/build/diffraction_functions.py") | buffer ~/Projects/diffraction_net/zernike3/build/diffraction_functions.py | else | edit ~/Projects/diffraction_net/zernike3/build/diffraction_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
140
normal! zo
let s:l = 148 - ((10 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
148
normal! 09|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/build/diffraction_functions.py") | buffer ~/Projects/diffraction_net/zernike3/build/diffraction_functions.py | else | edit ~/Projects/diffraction_net/zernike3/build/diffraction_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
198
normal! zo
let s:l = 205 - ((9 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
205
normal! 09|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//11136:/bin/bash") | buffer term://.//11136:/bin/bash | else | edit term://.//11136:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1 - ((0 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe '5resize ' . ((&lines * 7 + 24) / 49)
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
if bufexists("term://.//16116:/bin/bash") | buffer term://.//16116:/bin/bash | else | edit term://.//16116:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 372 - ((38 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
372
normal! 050|
lcd ~/Projects/diffraction_net
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
