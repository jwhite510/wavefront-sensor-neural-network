let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/projects/diffraction_network
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +3470 term://.//8162:/bin/bash
badd +149 term://.//8201:/bin/bash
badd +10 requirements.txt
badd +371 term://.//8449:/bin/bash
badd +192 _main.py
badd +14 run_tests.sh
badd +96 params.py
badd +156 diffraction_functions.py
badd +7 live_capture/TIS.py
badd +435 term://.//10775:/bin/bash
badd +2 something.txt
badd +715 term://.//17634:/bin/bash
badd +124 diffraction_net.py
badd +72 GetMeasuredDiffractionPattern.py
badd +496 term://.//27676:/bin/bash
badd +117 term://.//28951:/bin/bash
badd +0 test.py
badd +0 term://.//9541:/bin/bash
argglobal
%argdel
$argadd .
set stal=2
edit requirements.txt
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
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
wincmd _ | wincmd |
split
2wincmd k
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 6 + 27) / 54)
exe '2resize ' . ((&lines * 10 + 27) / 54)
exe 'vert 2resize ' . ((&columns * 152 + 102) / 204)
exe '3resize ' . ((&lines * 33 + 27) / 54)
exe 'vert 3resize ' . ((&columns * 152 + 102) / 204)
exe '4resize ' . ((&lines * 11 + 27) / 54)
exe 'vert 4resize ' . ((&columns * 51 + 102) / 204)
exe '5resize ' . ((&lines * 10 + 27) / 54)
exe 'vert 5resize ' . ((&columns * 51 + 102) / 204)
exe '6resize ' . ((&lines * 21 + 27) / 54)
exe 'vert 6resize ' . ((&columns * 51 + 102) / 204)
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 12 - ((4 * winheight(0) + 3) / 6)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
12
normal! 0
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("~/projects/diffraction_network/diffraction_functions.py") | buffer ~/projects/diffraction_network/diffraction_functions.py | else | edit ~/projects/diffraction_network/diffraction_functions.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
25,28fold
34,35fold
37,38fold
42,45fold
75,77fold
72,80fold
104,104fold
125,127fold
129,129fold
146,148fold
150,150fold
154,157fold
181,183fold
185,185fold
191,193fold
195,195fold
33,199fold
202,221fold
239,241fold
243,245fold
263,263fold
263,263fold
262,263fold
269,269fold
269,269fold
268,269fold
272,272fold
272,272fold
271,272fold
265,274fold
224,293fold
297,311fold
315,350fold
354,360fold
354,360fold
364,418fold
364,418fold
422,460fold
422,460fold
468,470fold
472,474fold
464,476fold
481,485fold
481,485fold
479,486fold
491,496fold
500,506fold
512,512fold
514,514fold
546,546fold
548,548fold
511,558fold
564,578fold
581,598fold
602,605fold
611,613fold
624,625fold
601,627fold
631,634fold
640,642fold
653,654fold
630,656fold
660,675fold
678,689fold
713,714fold
710,714fold
721,722fold
717,722fold
692,724fold
727,738fold
771,773fold
741,775fold
778,780fold
783,787fold
791,793fold
790,802fold
807,808fold
806,808fold
828,832fold
805,850fold
854,885fold
888,920fold
926,926fold
935,935fold
950,957fold
997,998fold
1000,1001fold
924,1003fold
1028,1062fold
33
normal! zo
511
normal! zo
let s:l = 737 - ((6 * winheight(0) + 5) / 10)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
737
normal! 05|
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("term://.//8162:/bin/bash") | buffer term://.//8162:/bin/bash | else | edit term://.//8162:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 10026 - ((25 * winheight(0) + 16) / 33)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
10026
normal! 0
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("~/projects/diffraction_network/test.py") | buffer ~/projects/diffraction_network/test.py | else | edit ~/projects/diffraction_network/test.py | endif
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
3
normal! zo
4
normal! zo
let s:l = 4 - ((3 * winheight(0) + 5) / 11)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
4
normal! 014|
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("term://.//28951:/bin/bash") | buffer term://.//28951:/bin/bash | else | edit term://.//28951:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 246 - ((9 * winheight(0) + 5) / 10)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
246
normal! 0
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("term://.//8449:/bin/bash") | buffer term://.//8449:/bin/bash | else | edit term://.//8449:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 374 - ((20 * winheight(0) + 10) / 21)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
374
normal! 0
lcd ~/projects/diffraction_network
wincmd w
exe '1resize ' . ((&lines * 6 + 27) / 54)
exe '2resize ' . ((&lines * 10 + 27) / 54)
exe 'vert 2resize ' . ((&columns * 152 + 102) / 204)
exe '3resize ' . ((&lines * 33 + 27) / 54)
exe 'vert 3resize ' . ((&columns * 152 + 102) / 204)
exe '4resize ' . ((&lines * 11 + 27) / 54)
exe 'vert 4resize ' . ((&columns * 51 + 102) / 204)
exe '5resize ' . ((&lines * 10 + 27) / 54)
exe 'vert 5resize ' . ((&columns * 51 + 102) / 204)
exe '6resize ' . ((&lines * 21 + 27) / 54)
exe 'vert 6resize ' . ((&columns * 51 + 102) / 204)
tabedit ~/projects/diffraction_network/GetMeasuredDiffractionPattern.py
set splitbelow splitright
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
exe '1resize ' . ((&lines * 24 + 27) / 54)
exe 'vert 1resize ' . ((&columns * 95 + 102) / 204)
exe '2resize ' . ((&lines * 26 + 27) / 54)
exe 'vert 2resize ' . ((&columns * 95 + 102) / 204)
exe '3resize ' . ((&lines * 5 + 27) / 54)
exe 'vert 3resize ' . ((&columns * 108 + 102) / 204)
exe '4resize ' . ((&lines * 45 + 27) / 54)
exe 'vert 4resize ' . ((&columns * 108 + 102) / 204)
argglobal
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
9
normal! zo
50
normal! zo
61
normal! zc
73
normal! zo
73
normal! zc
let s:l = 57 - ((8 * winheight(0) + 12) / 24)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
57
let s:c = 16 - ((11 * winwidth(0) + 47) / 95)
if s:c > 0
  exe 'normal! ' . s:c . '|zs' . 16 . '|'
else
  normal! 016|
endif
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("~/projects/diffraction_network/_main.py") | buffer ~/projects/diffraction_network/_main.py | else | edit ~/projects/diffraction_network/_main.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
18,23fold
17,23fold
106,110fold
105,110fold
122,123fold
122,123fold
29,135fold
138,139fold
143,143fold
152,152fold
148,154fold
157,158fold
147,158fold
161,163fold
167,169fold
172,173fold
166,173fold
175,176fold
179,182fold
175,182fold
204,204fold
207,207fold
210,210fold
213,213fold
193,260fold
191,260fold
266,266fold
278,280fold
282,282fold
265,282fold
28,282fold
295,313fold
17
normal! zo
28
normal! zo
29
normal! zo
175
normal! zo
191
normal! zo
193
normal! zo
265
normal! zo
let s:l = 311 - ((16 * winheight(0) + 13) / 26)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
311
let s:c = 63 - ((42 * winwidth(0) + 47) / 95)
if s:c > 0
  exe 'normal! ' . s:c . '|zs' . 63 . '|'
else
  normal! 063|
endif
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("~/projects/diffraction_network/diffraction_functions.py") | buffer ~/projects/diffraction_network/diffraction_functions.py | else | edit ~/projects/diffraction_network/diffraction_functions.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
25,28fold
34,35fold
37,38fold
42,45fold
75,77fold
72,80fold
104,104fold
125,127fold
129,129fold
146,148fold
150,150fold
154,157fold
181,183fold
185,185fold
191,193fold
195,195fold
33,199fold
202,221fold
239,241fold
243,245fold
263,263fold
263,263fold
262,263fold
269,269fold
269,269fold
268,269fold
272,272fold
272,272fold
271,272fold
265,274fold
224,293fold
297,311fold
315,350fold
354,360fold
354,360fold
364,418fold
364,418fold
422,460fold
422,460fold
468,470fold
472,474fold
464,476fold
481,485fold
481,485fold
479,486fold
491,496fold
500,506fold
512,512fold
514,514fold
546,546fold
548,548fold
511,558fold
564,578fold
581,598fold
602,605fold
611,613fold
624,625fold
601,627fold
631,634fold
640,642fold
653,654fold
630,656fold
660,675fold
678,689fold
713,714fold
710,714fold
721,722fold
717,722fold
692,724fold
727,738fold
771,773fold
741,775fold
778,780fold
783,787fold
791,793fold
790,802fold
807,808fold
806,808fold
828,832fold
805,850fold
854,885fold
888,920fold
926,926fold
935,935fold
950,957fold
997,998fold
1000,1001fold
924,1003fold
1028,1062fold
33
normal! zo
224
normal! zo
511
normal! zo
let s:l = 22 - ((3 * winheight(0) + 2) / 5)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
22
normal! 011|
lcd ~/projects/diffraction_network
wincmd w
argglobal
if bufexists("~/projects/diffraction_network/diffraction_functions.py") | buffer ~/projects/diffraction_network/diffraction_functions.py | else | edit ~/projects/diffraction_network/diffraction_functions.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
25,28fold
34,35fold
37,38fold
42,45fold
75,77fold
72,80fold
104,104fold
125,127fold
129,129fold
146,148fold
150,150fold
154,157fold
181,183fold
185,185fold
191,193fold
195,195fold
33,199fold
202,221fold
239,241fold
243,245fold
263,263fold
263,263fold
262,263fold
269,269fold
269,269fold
268,269fold
272,272fold
272,272fold
271,272fold
265,274fold
224,293fold
297,311fold
315,350fold
354,360fold
354,360fold
364,418fold
364,418fold
422,460fold
422,460fold
468,470fold
472,474fold
464,476fold
481,485fold
481,485fold
479,486fold
491,496fold
500,506fold
512,512fold
514,514fold
546,546fold
548,548fold
511,558fold
564,578fold
581,598fold
602,605fold
611,613fold
624,625fold
601,627fold
631,634fold
640,642fold
653,654fold
630,656fold
660,675fold
678,689fold
713,714fold
710,714fold
721,722fold
717,722fold
692,724fold
727,738fold
771,773fold
741,775fold
778,780fold
783,787fold
791,793fold
790,802fold
807,808fold
806,808fold
828,832fold
805,850fold
854,885fold
888,920fold
926,926fold
935,935fold
950,957fold
997,998fold
1000,1001fold
924,1003fold
1028,1062fold
33
normal! zo
224
normal! zo
239
normal! zc
243
normal! zc
262
normal! zo
262
normal! zc
265
normal! zo
265
normal! zc
511
normal! zo
let s:l = 249 - ((32 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
249
normal! 06|
lcd ~/projects/diffraction_network
wincmd w
3wincmd w
exe '1resize ' . ((&lines * 24 + 27) / 54)
exe 'vert 1resize ' . ((&columns * 95 + 102) / 204)
exe '2resize ' . ((&lines * 26 + 27) / 54)
exe 'vert 2resize ' . ((&columns * 95 + 102) / 204)
exe '3resize ' . ((&lines * 5 + 27) / 54)
exe 'vert 3resize ' . ((&columns * 108 + 102) / 204)
exe '4resize ' . ((&lines * 45 + 27) / 54)
exe 'vert 4resize ' . ((&columns * 108 + 102) / 204)
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
if bufexists("term://.//9541:/bin/bash") | buffer term://.//9541:/bin/bash | else | edit term://.//9541:/bin/bash | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=20
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 804 - ((50 * winheight(0) + 25) / 51)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
804
normal! 069|
lcd ~/projects/diffraction_network
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
