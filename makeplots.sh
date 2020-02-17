#!/bin/bash

dest_dir=$HOME/Documents/14_2_20_results/
declare -a run_names=(
"GHKWL_wavefrontsensor_with_phasesubtraction_gaussian1"
"GHKWL_wavefrontsensor_with_phasesubtraction_gaussian2"
"UIBKJBL_scalar_phase_withphasesubtract1"
"UIBKJBL_scalar_phase_withphasesubtract2"
)
for run_name in "${run_names[@]}"
do
	# delete the older tf events files if there are more than one
	items=($(ls -1v ~/ara_data/runs/tensorboard_graph/$run_name/events*))
	if ((${#items[@]} > 1))
	then
		# iterate through items
		for i in `seq 0 $((${#items[@]}-2))`
		do
			# echo $i
			rm ${items[$i]}
		done
	fi


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

	mkdir $dest_dir$run_name/pictures/${arr[-1]}
	cp -r ~/ara_data/runs/nn_pictures/$run_name\_pictures/${arr[-1]}/* $dest_dir$run_name/pictures/${arr[-1]}

done


