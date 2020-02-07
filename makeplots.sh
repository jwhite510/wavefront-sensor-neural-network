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
	mkdir $dest_dir$run_name/pictures/
	echo $run_name
	photo_folders=$(ls -1v ~/ara_data/runs/nn_pictures/$run_name\_pictures/)

	# convert string with whitespace to an array
	arr=($photo_folders)

	mkdir $dest_dir$run_name/pictures/${arr[-2]}
	cp -r ~/ara_data/runs/nn_pictures/$run_name\_pictures/${arr[-2]}/* $dest_dir$run_name/pictures/${arr[-2]}

done


