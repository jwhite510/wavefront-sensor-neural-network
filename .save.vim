let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Projects/diffraction_net
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +32 zernike3/CMakeLists.txt
badd +1 zernike3/src/MeasuredImageFormatter.cpp
badd +3 zernike3/src/MeasuredImageFormatterWrapper.cpp
badd +0 zernike3/build/MeasuredImageFormatterTest.py
badd +1 zernike3/src/MeasuredImageFormatter.h
badd +3 zernike3/src/MeasuredImageFormatterWrapper.h
badd +0 term://.//5766:/bin/bash
badd +0 run.sh
badd +0 rungdb.sh
badd +0 term://.//7144:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit zernike3/CMakeLists.txt
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
exe '1resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 126 + 106) / 212)
exe '4resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 126 + 106) / 212)
exe '5resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 85 + 106) / 212)
exe '6resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 85 + 106) / 212)
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
let s:l = 32 - ((8 * winheight(0) + 6) / 13)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
32
normal! 052|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("term://.//5766:/bin/bash") | buffer term://.//5766:/bin/bash | else | edit term://.//5766:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 130 - ((12 * winheight(0) + 6) / 13)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
130
normal! 065|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/MeasuredImageFormatterWrapper.cpp") | buffer ~/Projects/diffraction_net/zernike3/src/MeasuredImageFormatterWrapper.cpp | else | edit ~/Projects/diffraction_net/zernike3/src/MeasuredImageFormatterWrapper.cpp | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
7
normal! zo
8
normal! zo
12
normal! zo
let s:l = 13 - ((11 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
13
normal! 08|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/src/MeasuredImageFormatter.cpp") | buffer ~/Projects/diffraction_net/zernike3/src/MeasuredImageFormatter.cpp | else | edit ~/Projects/diffraction_net/zernike3/src/MeasuredImageFormatter.cpp | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
7
normal! zo
10
normal! zo
let s:l = 9 - ((4 * winheight(0) + 8) / 16)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
9
normal! 030|
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3/build/MeasuredImageFormatterTest.py") | buffer ~/Projects/diffraction_net/zernike3/build/MeasuredImageFormatterTest.py | else | edit ~/Projects/diffraction_net/zernike3/build/MeasuredImageFormatterTest.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 4 - ((3 * winheight(0) + 8) / 17)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/zernike3") | buffer ~/Projects/diffraction_net/zernike3 | else | edit ~/Projects/diffraction_net/zernike3 | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 10 - ((3 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
3wincmd w
exe '1resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 13 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 126 + 106) / 212)
exe '4resize ' . ((&lines * 16 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 126 + 106) / 212)
exe '5resize ' . ((&lines * 17 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 85 + 106) / 212)
exe '6resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 6resize ' . ((&columns * 85 + 106) / 212)
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
if bufexists("term://.//7144:/bin/bash") | buffer term://.//7144:/bin/bash | else | edit term://.//7144:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 171 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
171
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
