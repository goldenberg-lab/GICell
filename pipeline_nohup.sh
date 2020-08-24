#!/bin/bash

echo "start of shell"

conda activate QuPath

cell_inflam="eosinophil,neutrophil,plasma,lymphocyte"
cell_eosin="eosinophil"

echo "running eosin"
python -u script_mdl_cell.py --cells $cell_eosin --num_epochs 29 --epoch_check 29 --batch_size 12 --learning_rate 0.001 --num_params 16
echo "running inflam"
python -u script_mdl_cell.py --cells $cell_inflam --num_epochs 29 --epoch_check 29 --batch_size 12 --learning_rate 0.001 --num_params 16

echo "end of shell"
