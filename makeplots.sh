#!/bin/bash

dest_dir=$HOME/Documents/AHGNKL_run/
declare -a run_names=(
"AHGNKL_withphasesubtraction1"
"AHGNKL_withphasesubtraction2"
"AHGNKL_withOUTphasesubtraction1"
"AHGNKL_withOUTphasesubtraction2"
"AHGNKL_withphasesubtraction_increaseradius1"
"AHGNKL_withphasesubtraction_increaseradius2"
)
for run_name in "${run_names[@]}"
do
	python makeplots.py ~/ara_data/runs/tensorboard_graph/$run_name/
	mkdir $dest_dir
	mkdir $dest_dir$run_name/
	cp ~/ara_data/runs/tensorboard_graph/$run_name/csv/$run_name.png $dest_dir$run_name/
	# copy pictures

	# echo $run_name\_pictures
	mkdir $dest_dir$run_name/pictures/
	cp -r ~/ara_data/runs/nn_pictures/$run_name\_pictures/* $dest_dir$run_name/pictures/
done


