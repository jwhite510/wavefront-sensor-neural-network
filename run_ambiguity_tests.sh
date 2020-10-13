#!/bin/bash


# python ambiguitynet.py --wfsensor 0 --filename thename
names=(
'round_down_left_10x10_'
'round_down_left_10x10_random_offset_'
'square_6x6_up_right_'
'round_10x10_up_right_'
'square_10x10_up_right_'
)
for k in $(seq 0 4)
do
	echo $k
	echo ${names[$k]}
	python  ambiguitynet.py --wfsensor $k --filename ${names[$k]}
done


