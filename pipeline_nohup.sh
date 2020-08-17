#!/bin/bash

conda activate QuPath

nohup python -u script_mdl_cell.py --cells eosinophil,neutrophil,plasma,lymphocyte --num_epochs 500 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 50 > ../inflam.log &

#nohup python -u script_mdl_cell.py --cells eosinophil --num_epochs 500 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 50 > ../eosin.log &


