#!/bin/sh

run_name=phase_and_amplitude_logloss_1_long/
echo $run_name
python makeplots.py ~/ara_data/tensorboard_graph/$run_name
