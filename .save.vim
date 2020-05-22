let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Projects/diffraction_net
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 zernike3/CMakeLists.txt
badd +160 zernike3/src/MeasuredImageFormatter.cpp
badd +11 zernike3/src/MeasuredImageFormatterWrapper.cpp
badd +11 zernike3/build/MeasuredImageFormatterTest.py
badd +1 zernike3/src/MeasuredImageFormatter.h
badd +220 diffraction_net.py
badd +304 zernike3/src/zernikedatagen.h
badd +1 GetMeasuredDiffractionPattern.py
badd +140 diffraction_functions.py
badd +4 zernike3/src/MeasuredImageFormatterWrapper.h
badd +91 term://.//763:/bin/bash
badd +1 run.sh
badd +1 rungdb.sh
badd +937 term://.//7144:/bin/bash
badd +6 zernike3/src/c_arrays.h
badd +1 debug.gdb
badd +2 run_tests.sh
badd +904 term://.//10796:/bin/bash
badd +534 term://.//18391:/bin/bash
badd +1 zernike3/build/plot.gnu
badd +1 todo.txt
badd +12 zernike3/build/diffraction_functions.py
badd +1 zernike3/build/plot.sh
badd +114 term://.//898:/bin/bash
badd +46 term://.//13290:/bin/bash
badd +357 term://.//15313:/bin/bash
badd +1 something.txt
badd +1 term://.//711:/bin/bash
badd +520 term://.//27412:/bin/bash
badd +1 term://.//806:/bin/bash
badd +39 term://.//1666:/bin/bash
badd +1 term://.//968:/bin/bash
badd +668 term://.//13811:/bin/bash
badd +55 zernike3/src/testprop.cpp
badd +815 term://.//19963:/bin/bash
badd +408 term://.//24327:/bin/bash
badd +1 term://.//1017:/bin/bash
badd +11 subpixelshifttest.py
badd +0 term://.//8620:/bin/bash
badd +9 phase_normalization.py
argglobal
%argdel
$argadd ~/Projects/diffraction_net/
set stal=2
edit zernike3/CMakeLists.txt
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
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 33 - ((32 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
33
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//711:/bin/bash") | buffer term://.//711:/bin/bash | else | edit term://.//711:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 261 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
261
normal! 050|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/Projects/diffraction_net/subpixelshifttest.py
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
exe 'vert 1resize ' . ((&columns * 86 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 125 + 106) / 212)
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
8,33fold
7,33fold
7
normal! zo
8
normal! zo
let s:l = 16 - ((15 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
16
normal! 08|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//8620:/bin/bash") | buffer term://.//8620:/bin/bash | else | edit term://.//8620:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1945 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1945
normal! 050|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 86 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 125 + 106) / 212)
tabedit ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 5 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
exe '5resize ' . ((&lines * 12 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 106 + 106) / 212)
exe '6resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 105 + 106) / 212)
argglobal
if bufexists("term://.//763:/bin/bash") | buffer term://.//763:/bin/bash | else | edit term://.//763:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 352 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
352
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//806:/bin/bash") | buffer term://.//806:/bin/bash | else | edit term://.//806:/bin/bash | endif
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
lcd ~/Projects/diffraction_net
wincmd w
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
7
normal! zo
48
normal! zo
let s:l = 57 - ((6 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
57
normal! 013|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/diffraction_net.py") | buffer ~/Projects/diffraction_net/diffraction_net.py | else | edit ~/Projects/diffraction_net/diffraction_net.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
62
normal! zo
63
normal! zo
178
normal! zo
204
normal! zo
208
normal! zo
209
normal! zo
213
normal! zo
let s:l = 219 - ((0 * winheight(0) + 2) / 5)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
219
normal! 083|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/diffraction_functions.py") | buffer ~/Projects/diffraction_net/diffraction_functions.py | else | edit ~/Projects/diffraction_net/diffraction_functions.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 10 - ((1 * winheight(0) + 6) / 12)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10
normal! 012|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/build/MeasuredImageFormatterTest.py") | buffer ~/Projects/diffraction_net/zernike3/build/MeasuredImageFormatterTest.py | else | edit ~/Projects/diffraction_net/zernike3/build/MeasuredImageFormatterTest.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=2
setlocal fml=1
setlocal fdn=20
setlocal fen
25
normal! zo
let s:l = 19 - ((18 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
19
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 5 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 106 + 106) / 212)
exe '5resize ' . ((&lines * 12 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 106 + 106) / 212)
exe '6resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/Projects/diffraction_net/zernike3/src/MeasuredImageFormatter.cpp
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
exe 'vert 1resize ' . ((&columns * 124 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 87 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
68
normal! zo
77
normal! zo
80
normal! zo
88
normal! zo
89
normal! zo
98
normal! zo
111
normal! zo
122
normal! zo
129
normal! zo
149
normal! zo
154
normal! zo
154
normal! zc
162
normal! zo
172
normal! zo
174
normal! zc
188
normal! zo
204
normal! zo
188
normal! zo
196
normal! zo
196
normal! zc
202
normal! zo
202
normal! zc
225
normal! zo
225
normal! zc
231
normal! zo
231
normal! zc
246
normal! zo
253
normal! zo
261
normal! zo
let s:l = 149 - ((20 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
149
normal! 03|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//898:/bin/bash") | buffer term://.//898:/bin/bash | else | edit term://.//898:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 32 - ((31 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
32
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 124 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 87 + 106) / 212)
tabedit ~/Projects/diffraction_net/zernike3/build/plot.sh
set splitbelow splitright
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
argglobal
if bufexists("term://.//968:/bin/bash") | buffer term://.//968:/bin/bash | else | edit term://.//968:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 13 - ((8 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
13
normal! 065|
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
let s:l = 13 - ((12 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
13
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
tabedit ~/Projects/diffraction_net/diffraction_net.py
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
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
62
normal! zo
63
normal! zo
let s:l = 191 - ((22 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
191
normal! 017|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/diffraction_functions.py") | buffer ~/Projects/diffraction_net/diffraction_functions.py | else | edit ~/Projects/diffraction_net/diffraction_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
141
normal! zo
let s:l = 148 - ((8 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
148
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py") | buffer ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py | else | edit ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py | endif
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
12
normal! zo
39
normal! zo
48
normal! zo
56
normal! zo
64
normal! zo
65
normal! zo
68
normal! zo
71
normal! zo
74
normal! zo
let s:l = 55 - ((4 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
55
normal! 062|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/Projects/diffraction_net/todo.txt
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
normal! 017|
lcd ~/Projects/diffraction_net
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
