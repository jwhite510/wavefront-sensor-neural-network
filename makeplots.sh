#!/bin/sh

run_name=AHGNKL_withphasesubtraction_increaseradius1
dest_dir=$HOME/Documents/AHGNKL_run/

python makeplots.py ~/ara_data/runs/tensorboard_graph/$run_name/
mkdir $dest_dir
mkdir $dest_dir$run_name/
cp ~/ara_data/runs/tensorboard_graph/$run_name/csv/$run_name.png $dest_dir$run_name/
