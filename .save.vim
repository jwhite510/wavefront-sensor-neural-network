let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/diffraction_net
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +12 todo.txt
badd +5 todo_old.txt
badd +78 zernike3/src/main.cpp
badd +277 zernike3/src/pythonarrays.h
badd +254 zernike3/build/utility.py
badd +5 zernike3/runmpi.sh
badd +6 run_tests.sh
badd +551 zernike3/src/zernikedatagen.h
badd +323 diffraction_functions.py
badd +11 calibrate_measured_data.py
badd +26 diffraction_net.py
badd +72 zernike3/build/addnoise.py
badd +4 add_noise.sh
badd +44 zernike3/build/PropagateTF.py
badd +153 ~/.bashrc
badd +783 ~/.vimrc
badd +0 term://.//7464:/bin/bash
badd +58 term://.//9163:/bin/bash
badd +73 CompareNN_MatlabBilinearInterp.py
badd +33 PropagateSphericalAperture.py
badd +18 zernike3/build/testprop.py
badd +103 GetMeasuredDiffractionPattern.py
badd +1 zernike3/build/plot.py
badd +674 term://.//12063:/bin/bash
badd +485 term://.//27496:/bin/bash
badd +492 term://.//27735:/bin/bash
badd +698 term://.//37257:/bin/bash
badd +18 zernike3/src/utility.h
badd +0 term://.//43436:/bin/bash
badd +6 term://.//33644:/bin/bash
badd +67 term://.//4693:/bin/bash
badd +0 term://.//4892:/bin/bash
badd +24 zernike3/build/viewdata.py
badd +0 term://.//5651:/bin/bash
badd +0 term://.//17764:/bin/bash
argglobal
%argdel
$argadd ./
set stal=2
edit add_noise.sh
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
exe '1resize ' . ((&lines * 3 + 24) / 49)
exe '2resize ' . ((&lines * 42 + 24) / 49)
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
let s:l = 3 - ((0 * winheight(0) + 1) / 3)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
3
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
let s:l = 50 - ((6 * winheight(0) + 21) / 42)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
50
normal! 037|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 3 + 24) / 49)
exe '2resize ' . ((&lines * 42 + 24) / 49)
tabedit ~/diffraction_net/run_tests.sh
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
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
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 96 + 106) / 212)
exe '2resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 96 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 115 + 106) / 212)
exe '4resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 115 + 106) / 212)
exe '5resize ' . ((&lines * 7 + 24) / 49)
argglobal
if bufexists("term://.//7464:/bin/bash") | buffer term://.//7464:/bin/bash | else | edit term://.//7464:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 10011 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10011
normal! 039|
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
let s:l = 11 - ((2 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 0
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//9163:/bin/bash") | buffer term://.//9163:/bin/bash | else | edit term://.//9163:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 14 - ((1 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
14
normal! 014|
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
let s:l = 129 - ((5 * winheight(0) + 7) / 14)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
129
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//43436:/bin/bash") | buffer term://.//43436:/bin/bash | else | edit term://.//43436:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 49 - ((6 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
49
normal! 039|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 96 + 106) / 212)
exe '2resize ' . ((&lines * 26 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 96 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 115 + 106) / 212)
exe '4resize ' . ((&lines * 14 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 115 + 106) / 212)
exe '5resize ' . ((&lines * 7 + 24) / 49)
tabedit ~/diffraction_net/run_tests.sh
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
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
argglobal
if bufexists("term://.//4892:/bin/bash") | buffer term://.//4892:/bin/bash | else | edit term://.//4892:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 1062 - ((1 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1062
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
let s:l = 24 - ((23 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
24
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("term://.//4693:/bin/bash") | buffer term://.//4693:/bin/bash | else | edit term://.//4693:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 230 - ((37 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
230
normal! 054|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
tabedit ~/diffraction_net/zernike3/build/viewdata.py
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
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
argglobal
if bufexists("term://.//5651:/bin/bash") | buffer term://.//5651:/bin/bash | else | edit term://.//5651:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 103 - ((6 * winheight(0) + 3) / 7)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
103
normal! 054|
lcd ~/diffraction_net
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
8
normal! zo
18
normal! zo
let s:l = 20 - ((14 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
20
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_functions.py") | buffer ~/diffraction_net/diffraction_functions.py | else | edit ~/diffraction_net/diffraction_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal nofen
29
normal! zo
320
normal! zo
836
normal! zo
850
normal! zo
let s:l = 331 - ((13 * winheight(0) + 19) / 38)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
331
normal! 09|
lcd ~/diffraction_net
wincmd w
3wincmd w
exe '1resize ' . ((&lines * 7 + 24) / 49)
exe '2resize ' . ((&lines * 38 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 38 + 24) / 49)
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
if bufexists("term://.//17764:/bin/bash") | buffer term://.//17764:/bin/bash | else | edit term://.//17764:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 457 - ((45 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
457
normal! 039|
lcd ~/diffraction_net
tabedit ~/diffraction_net/zernike3/src/main.cpp
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
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
38
normal! zo
89
normal! zo
103
normal! zo
109
normal! zo
116
normal! zo
120
normal! zo
let s:l = 79 - ((31 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
79
normal! 03|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/diffraction_functions.py") | buffer ~/diffraction_net/diffraction_functions.py | else | edit ~/diffraction_net/diffraction_functions.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal nofen
29
normal! zo
320
normal! zo
836
normal! zo
850
normal! zo
let s:l = 326 - ((8 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
326
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/build/utility.py") | buffer ~/diffraction_net/zernike3/build/utility.py | else | edit ~/diffraction_net/zernike3/build/utility.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 252 - ((0 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
252
normal! 05|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/zernikedatagen.h") | buffer ~/diffraction_net/zernike3/src/zernikedatagen.h | else | edit ~/diffraction_net/zernike3/src/zernikedatagen.h | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=6
setlocal fml=1
setlocal fdn=20
setlocal fen
469
normal! zo
let s:l = 476 - ((10 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
476
normal! 05|
lcd ~/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 70 + 106) / 212)
exe 'vert 2resize ' . ((&columns * 70 + 106) / 212)
exe '3resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 70 + 106) / 212)
exe '4resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 4resize ' . ((&columns * 70 + 106) / 212)
tabedit ~/diffraction_net/diffraction_functions.py
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
argglobal
if bufexists("term://.//27735:/bin/bash") | buffer term://.//27735:/bin/bash | else | edit term://.//27735:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 911 - ((10 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
911
normal! 039|
lcd ~/diffraction_net
wincmd w
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=3
setlocal fml=1
setlocal fdn=20
setlocal nofen
29
normal! zo
320
normal! zo
836
normal! zo
850
normal! zo
let s:l = 327 - ((9 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
327
normal! 09|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/build/utility.py") | buffer ~/diffraction_net/zernike3/build/utility.py | else | edit ~/diffraction_net/zernike3/build/utility.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 449 - ((8 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
449
normal! 0
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 105 + 106) / 212)
exe '3resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
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
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
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
let s:l = 579 - ((30 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
579
normal! 05|
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
let s:l = 643 - ((7 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
643
normal! 05|
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
let s:l = 664 - ((7 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
664
normal! 09|
lcd ~/diffraction_net
wincmd w
exe 'vert 1resize ' . ((&columns * 105 + 106) / 212)
exe '2resize ' . ((&lines * 23 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe '3resize ' . ((&lines * 22 + 24) / 49)
exe 'vert 3resize ' . ((&columns * 106 + 106) / 212)
tabedit ~/diffraction_net/zernike3/src/zernikedatagen.h
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
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
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
336
normal! zo
344
normal! zo
358
normal! zo
359
normal! zo
363
normal! zo
364
normal! zo
365
normal! zo
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
let s:l = 365 - ((4 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
365
normal! 09|
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
let s:l = 688 - ((16 * winheight(0) + 17) / 34)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
688
normal! 07|
lcd ~/diffraction_net
wincmd w
argglobal
if bufexists("~/diffraction_net/zernike3/src/main.cpp") | buffer ~/diffraction_net/zernike3/src/main.cpp | else | edit ~/diffraction_net/zernike3/src/main.cpp | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=5
setlocal fml=1
setlocal fdn=20
setlocal fen
38
normal! zo
103
normal! zo
109
normal! zo
116
normal! zo
120
normal! zo
let s:l = 89 - ((20 * winheight(0) + 23) / 46)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
89
normal! 07|
lcd ~/diffraction_net
wincmd w
exe '1resize ' . ((&lines * 11 + 24) / 49)
exe 'vert 1resize ' . ((&columns * 106 + 106) / 212)
exe '2resize ' . ((&lines * 34 + 24) / 49)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 212)
exe 'vert 3resize ' . ((&columns * 105 + 106) / 212)
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
