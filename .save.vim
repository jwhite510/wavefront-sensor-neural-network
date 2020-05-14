let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Projects/diffraction_net
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +30 run_tests.sh
badd +0 term://.//2562:/bin/bash
badd +9 term://.//2919:/bin/bash
badd +0 term://.//3267:/bin/bash
badd +218 diffraction_net.py
badd +97 zernike3/src/main.cpp
badd +290 zernike3/src/zernikedatagen.h
badd +0 todo.txt
badd +19 diffraction_functions.py
badd +2 term://.//4630:/bin/bash
badd +0 term://.//4780:/bin/bash
badd +0 term://.//4842:/bin/bash
badd +0 .git/index
argglobal
%argdel
$argadd .
set stal=2
edit run_tests.sh
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
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 15 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
exe '3resize ' . ((&lines * 39 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 105 + 105) / 211)
argglobal
if bufexists("term://.//2562:/bin/bash") | buffer term://.//2562:/bin/bash | else | edit term://.//2562:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1798 - ((54 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1798
normal! 050|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//3267:/bin/bash") | buffer term://.//3267:/bin/bash | else | edit term://.//3267:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 20 - ((14 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
20
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
let s:l = 12 - ((11 * winheight(0) + 19) / 39)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
12
normal! 02|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 15 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
exe '3resize ' . ((&lines * 39 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 105 + 105) / 211)
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
exe 'vert 1resize ' . ((&columns * 135 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 75 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 75 + 105) / 211)
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
233
normal! zo
239
normal! zo
249
normal! zo
258
normal! zo
292
normal! zo
301
normal! zo
let s:l = 235 - ((35 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
235
normal! 09|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//4780:/bin/bash") | buffer term://.//4780:/bin/bash | else | edit term://.//4780:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 7 - ((6 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
7
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//4842:/bin/bash") | buffer term://.//4842:/bin/bash | else | edit term://.//4842:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 32 - ((4 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
32
normal! 09|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 135 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 75 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 75 + 105) / 211)
tabedit ~/Projects/diffraction_net/diffraction_net.py
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
233
normal! zo
239
normal! zo
249
normal! zo
258
normal! zo
292
normal! zo
301
normal! zo
let s:l = 218 - ((25 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
218
normal! 021|
lcd ~/Projects/diffraction_net
tabedit ~/Projects/diffraction_net/.git/index
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
exe '1resize ' . ((&lines * 27 + 29) / 58)
exe '2resize ' . ((&lines * 27 + 29) / 58)
argglobal
setlocal fdm=syntax
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=1
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 11 - ((10 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 03|
lcd ~/Projects/diffraction_net
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
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 27 + 29) / 58)
exe '2resize ' . ((&lines * 27 + 29) / 58)
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
let s:l = 4 - ((3 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 09|
lcd ~/Projects/diffraction_net
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
if bufexists("term://.//2919:/bin/bash") | buffer term://.//2919:/bin/bash | else | edit term://.//2919:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 4 - ((3 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 026|
lcd ~/Projects/diffraction_net
tabedit ~/Projects/diffraction_net/zernike3/src/main.cpp
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
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 105 + 105) / 211)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
32
normal! zo
95
normal! zo
let s:l = 96 - ((46 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
96
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
501
normal! zo
597
normal! zo
604
normal! zo
let s:l = 605 - ((9 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
605
normal! 013|
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
501
normal! zo
597
normal! zo
604
normal! zo
623
normal! zo
let s:l = 622 - ((6 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
622
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 105 + 105) / 211)
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
exe 'vert 1resize ' . ((&columns * 75 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 135 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 135 + 105) / 211)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
62
normal! zo
448
normal! zo
450
normal! zo
511
normal! zo
704
normal! zo
706
normal! zo
714
normal! zo
714
normal! zo
716
normal! zo
716
normal! zo
718
normal! zo
718
normal! zo
721
normal! zo
721
normal! zo
let s:l = 718 - ((97 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
718
normal! 035|
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
209
normal! zo
213
normal! zo
213
normal! zo
213
normal! zo
213
normal! zo
633
normal! zo
704
normal! zo
706
normal! zo
716
normal! zo
718
normal! zo
let s:l = 433 - ((13 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
433
normal! 09|
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
448
normal! zo
450
normal! zo
let s:l = 525 - ((19 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
525
normal! 017|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 75 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 135 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 3resize ' . ((&columns * 135 + 105) / 211)
tabedit ~/Projects/diffraction_net/diffraction_net.py
set splitbelow splitright
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
exe '1resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
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
178
normal! zo
204
normal! zo
208
normal! zo
209
normal! zo
209
normal! zo
213
normal! zo
213
normal! zo
213
normal! zo
213
normal! zo
633
normal! zo
704
normal! zo
706
normal! zo
716
normal! zo
718
normal! zo
let s:l = 109 - ((11 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
109
normal! 076|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/diffraction_functions.py") | buffer ~/Projects/diffraction_net/diffraction_functions.py | else | edit ~/Projects/diffraction_net/diffraction_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 374 - ((5 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
374
normal! 0
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
209
normal! zo
213
normal! zo
213
normal! zo
213
normal! zo
213
normal! zo
633
normal! zo
704
normal! zo
706
normal! zo
716
normal! zo
718
normal! zo
let s:l = 434 - ((14 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
434
normal! 09|
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe '2resize ' . ((&lines * 27 + 29) / 58)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
exe '3resize ' . ((&lines * 27 + 29) / 58)
tabedit ~/Projects/diffraction_net/zernike3/src/main.cpp
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
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
32
normal! zo
95
normal! zo
let s:l = 97 - ((27 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
97
normal! 019|
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
501
normal! zo
597
normal! zo
604
normal! zo
let s:l = 639 - ((35 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
639
normal! 05|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 105 + 105) / 211)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 211)
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
