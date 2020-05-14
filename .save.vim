let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Projects/diffraction_net/zernike3
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +97 src/main.cpp
badd +1 CMakeLists.txt
badd +1 ~/Projects/diffraction_net/run_tests.sh
badd +1 src/testprop.cpp
badd +46 term://.//10858:/bin/bash
badd +289 src/zernikedatagen.h
badd +1 run.sh
badd +457 /usr/include/fftw3.h
badd +57 build/testprop.py
badd +252 build/utility.py
badd +46 term://.//9648:/bin/bash
badd +46 term://.//9640:/bin/bash
badd +6 src/python.txt
badd +1103 term://.//17293:/bin/bash
badd +205 build/diffraction_functions.py
badd +5 build/convert_svg_to_png.py
badd +12 ~/Projects/diffraction_net/.gitignore
badd +1 term://.//27130:/bin/bash
badd +1 ~/Projects/diffraction_net/run.sh
badd +5 term://.//7063:/bin/bash
badd +6 ~/Projects/diffraction_net/testpropagation.py
badd +46 term://.//11188:/bin/bash
badd +89 src/pythonarrays.h
badd +93 src/c_arrays.h
badd +25 ~/.vimrc
badd +283 term://.//28056:/bin/bash
badd +2 ~/.editrc
badd +2 ~/Projects/diffraction_net/debug.gdb
badd +1 ~/Projects/diffraction_net/.save.vim
badd +161 term://.//15771:/bin/bash
badd +171 term://.//2613:/bin/bash
badd +1 ~/Projects/diffraction_net
badd +1 ~/Projects/diffraction_net/patch
badd +1 ~/Projects/diffraction_net/rungdb.sh
badd +5 ~/Projects/diffraction_net/todo.txt
badd +688 ~/Projects/diffraction_net/diffraction_net.py
badd +28 ~/Projects/diffraction_net/diffraction_functions.py
badd +159 ~/Projects/diffraction_net/calibrate_measured_data.py
badd +1 ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py
badd +19 term://.//7944:/bin/bash
badd +76 ~/Projects/diffraction_net/run_network.py
badd +156 term://.//18732:/bin/bash
badd +46 term://.//20265:/bin/bash
badd +1 ~/Projects/diffraction_net/.git/index
badd +0 term://.//7920:/bin/bash
badd +0 fugitive:///home/jonathon/Projects/diffraction_net/.git//0/.gitignore
argglobal
%argdel
$argadd build/
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
if bufexists("term://.//2613:/bin/bash") | buffer term://.//2613:/bin/bash | else | edit term://.//2613:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 46 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
46
normal! 0
lcd ~/Projects/diffraction_net
tabedit ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py
set splitbelow splitright
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
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 105 - ((21 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
105
normal! 05|
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
697
normal! zo
711
normal! zo
711
normal! zo
719
normal! zo
719
normal! zo
724
normal! zo
727
normal! zo
730
normal! zo
734
normal! zo
737
normal! zo
740
normal! zo
740
normal! zo
747
normal! zo
747
normal! zo
let s:l = 705 - ((9 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
705
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py") | buffer ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py | else | edit ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py | endif
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
64
normal! zo
let s:l = 72 - ((25 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
72
normal! 017|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py
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
7
normal! zo
48
normal! zo
let s:l = 57 - ((10 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
57
normal! 09|
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
138
normal! zo
188
normal! zo
697
normal! zo
711
normal! zo
711
normal! zo
719
normal! zo
719
normal! zo
724
normal! zo
727
normal! zo
730
normal! zo
734
normal! zo
737
normal! zo
740
normal! zo
740
normal! zo
747
normal! zo
747
normal! zo
let s:l = 137 - ((0 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
137
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/Projects/diffraction_net/GetMeasuredDiffractionPattern.py
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
setlocal fdl=4
setlocal fml=1
setlocal fdn=20
setlocal fen
7
normal! zo
48
normal! zo
64
normal! zo
let s:l = 104 - ((26 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
104
normal! 093|
lcd ~/Projects/diffraction_net
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
exe '1resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 146 + 106) / 212)
exe '2resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 65 + 106) / 212)
exe '3resize ' . ((&lines * 10 + 24) / 49)
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
180
normal! zo
186
normal! zo
202
normal! zo
let s:l = 195 - ((18 * winheight(0) + 17) / 35)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
195
normal! 0
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
697
normal! zo
let s:l = 21 - ((15 * winheight(0) + 17) / 35)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
21
normal! 016|
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
exe '1resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 146 + 106) / 212)
exe '2resize ' . ((&lines * 35 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 65 + 106) / 212)
exe '3resize ' . ((&lines * 10 + 24) / 49)
tabedit ~/Projects/diffraction_net/diffraction_net.py
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
setlocal fdl=13
setlocal fml=1
setlocal fdn=20
setlocal fen
62
normal! zo
63
normal! zo
180
normal! zo
186
normal! zo
202
normal! zo
let s:l = 189 - ((20 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
189
normal! 017|
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
180
normal! zo
186
normal! zo
202
normal! zo
208
normal! zo
261
normal! zo
265
normal! zo
270
normal! zo
let s:l = 267 - ((12 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
267
normal! 013|
lcd ~/Projects/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/Projects/diffraction_net/.git/index
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
exe '1resize ' . ((&lines * 19 + 24) / 49)
exe '2resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 26 + 24) / 49)
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
let s:l = 9 - ((8 * winheight(0) + 9) / 19)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
9
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("fugitive:///home/jonathon/Projects/diffraction_net/.git//0/.gitignore") | buffer fugitive:///home/jonathon/Projects/diffraction_net/.git//0/.gitignore | else | edit fugitive:///home/jonathon/Projects/diffraction_net/.git//0/.gitignore | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1 - ((0 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
argglobal
if bufexists("~/Projects/diffraction_net/.gitignore") | buffer ~/Projects/diffraction_net/.gitignore | else | edit ~/Projects/diffraction_net/.gitignore | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1 - ((0 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
lcd ~/Projects/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 19 + 24) / 49)
exe '2resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
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
if bufexists("term://.//7920:/bin/bash") | buffer term://.//7920:/bin/bash | else | edit term://.//7920:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 26 - ((25 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
26
normal! 050|
lcd ~/Projects/diffraction_net
tabnext 6
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
