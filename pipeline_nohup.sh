#!/bin/bash

echo "start of shell"

conda activate QuPath

cell_inflam="eosinophil,neutrophil,plasma,lymphocyte"
cell_eosin="eosinophil"

echo "running inflam"
python -u script_mdl_cell.py --cells $cell_inflam --num_epochs 60 --epoch_check 15 --batch_size 6 --learning_rate 0.001 --num_params 24

echo "running eosin"
python -u script_mdl_cell.py --cells $cell_eosin --num_epochs 60 --epoch_check 15 --batch_size 6 --learning_rate 0.001 --num_params 24

echo "end of shell"
