#!/bin/sh

run_name=AHGNKL_withphasesubtraction_increaseradius2

dest_dir=$HOME/Documents/AHGNKL_run/

python makeplots.py ~/ara_data/runs/tensorboard_graph/$run_name/
mkdir $dest_dir
mkdir $dest_dir$run_name/
cp ~/ara_data/runs/tensorboard_graph/$run_name/csv/$run_name.png $dest_dir$run_name/
# copy pictures

# echo $run_name\_pictures
mkdir $dest_dir$run_name/pictures/
cp -r ~/ara_data/runs/nn_pictures/$run_name\_pictures/* $dest_dir$run_name/pictures/
