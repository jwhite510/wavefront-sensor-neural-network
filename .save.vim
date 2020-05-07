let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Projects/diffraction_net/zernike3
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +7 src/main.cpp
badd +1 CMakeLists.txt
badd +1 ~/Projects/diffraction_net/run_tests.sh
badd +51 src/testprop.cpp
badd +1 term://.//26950:/bin/bash
badd +308 src/zernikedatagen.h
badd +1 run.sh
badd +457 /usr/include/fftw3.h
badd +61 build/testprop.py
badd +257 build/utility.py
badd +46 term://.//9648:/bin/bash
badd +46 term://.//9640:/bin/bash
badd +6 src/python.txt
badd +1 term://.//27031:/bin/bash
badd +1103 term://.//17293:/bin/bash
badd +150 build/diffraction_functions.py
badd +5 build/convert_svg_to_png.py
badd +1 ~/Projects/diffraction_net/.gitignore
badd +1 term://.//27130:/bin/bash
badd +1 ~/Projects/diffraction_net/run.sh
badd +0 term://.//27732:/bin/bash
badd +38 term://.//31012:/bin/bash
badd +6 ~/Projects/diffraction_net/testpropagation.py
badd +0 term://.//4831:/bin/bash
argglobal
%argdel
$argadd build/
set stal=2
edit CMakeLists.txt
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
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
argglobal
if bufexists("term://.//26950:/bin/bash") | buffer term://.//26950:/bin/bash | else | edit term://.//26950:/bin/bash | endif
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
lcd ~/Projects/diffraction_net
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
let s:l = 27 - ((26 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
27
normal! 023|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/testprop.cpp") | buffer ~/Projects/diffraction_net/zernike3/src/testprop.cpp | else | edit ~/Projects/diffraction_net/zernike3/src/testprop.cpp | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=2
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 59 - ((17 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
59
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/main.cpp") | buffer ~/Projects/diffraction_net/zernike3/src/main.cpp | else | edit ~/Projects/diffraction_net/zernike3/src/main.cpp | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 3 - ((2 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
3
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/Projects/diffraction_net/zernike3/build/testprop.py
set splitbelow splitright
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
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 31 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe '5resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 70 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=1
setlocal fml=1
setlocal fdn=20
setlocal fen
55
normal! zo
55
normal! zo
57
normal! zo
57
normal! zo
61
normal! zo
62
normal! zo
63
normal! zo
64
normal! zo
67
normal! zo
67
normal! zo
67
normal! zo
67
normal! zo
let s:l = 46 - ((37 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
46
normal! 09|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/Projects/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/Projects/diffraction_net/zernike3/src/zernikedatagen.h | endif
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
467
normal! zo
499
normal! zo
524
normal! zo
539
normal! zo
595
normal! zo
621
normal! zo
let s:l = 285 - ((13 * winheight(0) + 15) / 31)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
285
normal! 07|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/Projects/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/Projects/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
467
normal! zo
499
normal! zo
524
normal! zo
539
normal! zo
595
normal! zo
621
normal! zo
let s:l = 625 - ((5 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
625
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//27031:/bin/bash") | buffer term://.//27031:/bin/bash | else | edit term://.//27031:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 299 - ((6 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
299
normal! 0
lcd ~/Projects/diffraction_net/zernike3
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
let s:l = 53 - ((11 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
53
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe '2resize ' . ((&lines * 31 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
exe '5resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/Projects/diffraction_net/zernike3/build/testprop.py
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
exe '1resize ' . ((&lines * 36 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 28 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 9 + 24) / 49)
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
13
normal! zo
21
normal! zo
26
normal! zo
50
normal! zo
53
normal! zo
54
normal! zo
61
normal! zo
62
normal! zo
63
normal! zo
64
normal! zo
67
normal! zo
71
normal! zo
85
normal! zo
103
normal! zo
let s:l = 100 - ((16 * winheight(0) + 18) / 36)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
100
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/Projects/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/Projects/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
275
normal! zo
276
normal! zo
284
normal! zo
285
normal! zo
301
normal! zo
306
normal! zo
307
normal! zo
309
normal! zo
313
normal! zo
let s:l = 284 - ((4 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
284
normal! 07|
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
let s:l = 36 - ((9 * winheight(0) + 14) / 28)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
36
normal! 03|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//31012:/bin/bash") | buffer term://.//31012:/bin/bash | else | edit term://.//31012:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 107 - ((8 * winheight(0) + 4) / 9)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
107
normal! 065|
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 36 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 7 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 28 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 9 + 24) / 49)
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
let s:l = 5 - ((1 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
5
normal! $
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
13
normal! zo
21
normal! zo
26
normal! zo
50
normal! zo
53
normal! zo
54
normal! zo
61
normal! zo
62
normal! zo
63
normal! zo
64
normal! zo
67
normal! zo
71
normal! zo
85
normal! zo
103
normal! zo
let s:l = 54 - ((14 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
54
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
let s:l = 150 - ((12 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
150
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
let s:l = 205 - ((9 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
205
normal! 09|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//27732:/bin/bash") | buffer term://.//27732:/bin/bash | else | edit term://.//27732:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 40 - ((6 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
40
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
3wincmd w
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
if bufexists("term://.//4831:/bin/bash") | buffer term://.//4831:/bin/bash | else | edit term://.//4831:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 145 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
145
normal! 050|
lcd ~/Projects/diffraction_net
tabnext 4
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
