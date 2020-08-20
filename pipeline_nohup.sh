#!/bin/bash

conda activate QuPath

cell_inflam="eosinophil,neutrophil,plasma,lymphocyte"
cell_eosin="eosinophil"

#python script_mdl_cell.py --cells $cell_eosin --num_epochs 450 --epoch_check 150 --batch_size 1 --learning_rate 0.0005 --num_params 32
python script_mdl_cell.py --cells $cell_inflam --num_epochs 450 --epoch_check 150 --batch_size 1 --learning_rate 0.0005 --num_params 32

echo "end of shell"
