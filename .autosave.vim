let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Projects/diffraction
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +10 diffraction.py
badd +3 term://.//31496:/bin/bash
badd +8 term://.//32065:/bin/bash
argglobal
%argdel
$argadd .
edit diffraction.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
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
exe 'vert 1resize ' . ((&columns * 79 + 119) / 239)
exe 'vert 2resize ' . ((&columns * 79 + 119) / 239)
exe '3resize ' . ((&lines * 27 + 28) / 57)
exe 'vert 3resize ' . ((&columns * 79 + 119) / 239)
exe '4resize ' . ((&lines * 27 + 28) / 57)
exe 'vert 4resize ' . ((&columns * 79 + 119) / 239)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
96
normal! zo
let s:l = 129 - ((34 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
129
normal! 05|
lcd ~/Projects/diffraction
wincmd w
argglobal
if bufexists("~/Projects/diffraction/diffraction.py") | buffer ~/Projects/diffraction/diffraction.py | else | edit ~/Projects/diffraction/diffraction.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
96
normal! zo
let s:l = 302 - ((28 * winheight(0) + 27) / 55)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
302
normal! 0
lcd ~/Projects/diffraction
wincmd w
argglobal
if bufexists("term://.//32065:/bin/bash") | buffer term://.//32065:/bin/bash | else | edit term://.//32065:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 2415 - ((26 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2415
normal! 046|
lcd ~/Projects/diffraction
wincmd w
argglobal
if bufexists("~/Projects/diffraction/diffraction.py") | buffer ~/Projects/diffraction/diffraction.py | else | edit ~/Projects/diffraction/diffraction.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
96
normal! zo
let s:l = 21 - ((10 * winheight(0) + 13) / 27)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
21
normal! 06|
lcd ~/Projects/diffraction
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 79 + 119) / 239)
exe 'vert 2resize ' . ((&columns * 79 + 119) / 239)
exe '3resize ' . ((&lines * 27 + 28) / 57)
exe 'vert 3resize ' . ((&columns * 79 + 119) / 239)
exe '4resize ' . ((&lines * 27 + 28) / 57)
exe 'vert 4resize ' . ((&columns * 79 + 119) / 239)
tabnext 1
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
