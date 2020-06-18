let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +9 todo.txt
badd +1 todo_old.txt
badd +33 zernike3/src/main.cpp
badd +278 zernike3/src/pythonarrays.h
badd +43 zernike3/build/utility.py
badd +10008 term://.//45556:/bin/bash
badd +5 zernike3/runmpi.sh
badd +25 run_tests.sh
badd +538 zernike3/src/zernikedatagen.h
badd +205 diffraction_functions.py
badd +20 diffraction_net.py
badd +22 term://.//20302:/bin/bash
badd +1 .git/index
badd +385 term://.//20502:/bin/bash
badd +0 term://.//20791:/bin/bash
badd +2843 term://.//37544:/bin/bash
badd +36 zernike3/build/addnoise.py
badd +4 add_noise.sh
badd +1 term://.//44787:/bin/bash
badd +1 fugitive:///home/jonathon/diffraction_net/.git//0/zernike3/build/addnoise.py
badd +786 term://.//657:/bin/bash
badd +0 term://.//783:/bin/bash
badd +47 zernike3/build/PropagateTF.py
badd +153 ~/.bashrc
badd +0 term://.//6602:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit add_noise.sh
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
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 110 + 106) / 212)
exe '4resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 101 + 106) / 212)
argglobal
if bufexists("term://.//44787:/bin/bash") | buffer term://.//44787:/bin/bash | else | edit term://.//44787:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 333 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
333
normal! 0
lcd ~/diffraction_net
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
let s:l = 4 - ((2 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/build/addnoise.py") | buffer ~/diffraction_net/zernike3/build/addnoise.py | else | edit ~/diffraction_net/zernike3/build/addnoise.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=2
setlocal fml=1
setlocal fdn=20
setlocal fen
12
normal! zo
26
normal! zo
45
normal! zo
47
normal! zo
52
normal! zo
53
normal! zo
let s:l = 42 - ((16 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
42
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//783:/bin/bash") | buffer term://.//783:/bin/bash | else | edit term://.//783:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 2 - ((1 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2
normal! 012|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 110 + 106) / 212)
exe '4resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 101 + 106) / 212)
tabedit ~/diffraction_net/run_tests.sh
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
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 52 + 106) / 212)
argglobal
if bufexists("term://.//45556:/bin/bash") | buffer term://.//45556:/bin/bash | else | edit term://.//45556:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 10015 - ((14 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10015
normal! 053|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//6602:/bin/bash") | buffer term://.//6602:/bin/bash | else | edit term://.//6602:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 2 - ((1 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2
normal! 013|
lcd ~/diffraction_net
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
let s:l = 25 - ((13 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
25
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/build/addnoise.py") | buffer ~/diffraction_net/zernike3/build/addnoise.py | else | edit ~/diffraction_net/zernike3/build/addnoise.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
26
normal! zo
45
normal! zo
47
normal! zo
52
normal! zo
53
normal! zo
let s:l = 41 - ((35 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
41
normal! 017|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_net.py") | buffer ~/diffraction_net/diffraction_net.py | else | edit ~/diffraction_net/diffraction_net.py | endif
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
let s:l = 864 - ((10 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
864
normal! 023|
lcd ~/diffraction_net
wincmd w
3wincmd w
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 15 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
exe '4resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 52 + 106) / 212)
exe '5resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 5resize ' . ((&columns * 52 + 106) / 212)
tabedit ~/diffraction_net/.git/index
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
let s:l = 11 - ((8 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("fugitive:///home/jonathon/diffraction_net/.git//0/zernike3/build/addnoise.py") | buffer fugitive:///home/jonathon/diffraction_net/.git//0/zernike3/build/addnoise.py | else | edit fugitive:///home/jonathon/diffraction_net/.git//0/zernike3/build/addnoise.py | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 52 - ((51 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
52
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/build/addnoise.py") | buffer ~/diffraction_net/zernike3/build/addnoise.py | else | edit ~/diffraction_net/zernike3/build/addnoise.py | endif
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 52 - ((51 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
52
normal! 0
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 15 + 24) / 49)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/zernike3/src/zernikedatagen.h
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
exe 'vert 1resize ' . ((&columns * 86 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 125 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 125 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
501
normal! zo
526
normal! zo
526
normal! zo
531
normal! zo
541
normal! zo
573
normal! zo
574
normal! zo
582
normal! zo
588
normal! zo
589
normal! zo
601
normal! zo
629
normal! zo
638
normal! zo
641
normal! zo
645
normal! zo
649
normal! zo
655
normal! zo
656
normal! zo
664
normal! zo
664
normal! zo
678
normal! zo
683
normal! zo
684
normal! zo
688
normal! zo
691
normal! zo
695
normal! zo
702
normal! zo
704
normal! zo
711
normal! zo
let s:l = 580 - ((21 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
580
normal! 03|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
501
normal! zo
526
normal! zo
526
normal! zo
531
normal! zo
541
normal! zo
573
normal! zo
574
normal! zo
582
normal! zo
588
normal! zo
589
normal! zo
601
normal! zo
629
normal! zo
638
normal! zo
641
normal! zo
645
normal! zo
649
normal! zo
655
normal! zo
656
normal! zo
664
normal! zo
664
normal! zo
678
normal! zo
683
normal! zo
684
normal! zo
688
normal! zo
691
normal! zo
695
normal! zo
702
normal! zo
704
normal! zo
711
normal! zo
let s:l = 647 - ((10 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
647
normal! 013|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
501
normal! zo
526
normal! zo
526
normal! zo
531
normal! zo
541
normal! zo
573
normal! zo
574
normal! zo
582
normal! zo
588
normal! zo
589
normal! zo
601
normal! zo
629
normal! zo
638
normal! zo
641
normal! zo
645
normal! zo
649
normal! zo
655
normal! zo
656
normal! zo
664
normal! zo
664
normal! zo
678
normal! zo
683
normal! zo
684
normal! zo
688
normal! zo
691
normal! zo
695
normal! zo
702
normal! zo
704
normal! zo
711
normal! zo
let s:l = 665 - ((8 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
665
normal! 011|
lcd ~/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 86 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 125 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 125 + 106) / 212)
tabedit ~/diffraction_net/todo.txt
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
exe '1resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
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
let s:l = 12 - ((9 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
12
normal! 02|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/todo_old.txt") | buffer ~/diffraction_net/todo_old.txt | else | edit ~/diffraction_net/todo_old.txt | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 5 - ((2 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
5
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//20791:/bin/bash") | buffer term://.//20791:/bin/bash | else | edit term://.//20791:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 96 - ((14 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
96
normal! 039|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 30 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 15 + 24) / 49)
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
