#!/bin/bash

echo "start of shell"

#conda activate QuPath

cell_inflam="eosinophil,neutrophil,plasma,lymphocyte"
cell_eosin="eosinophil"

# Loop over different learning rate/batch_size/architecture configurations

num_epochs=90
epoch_check=15
lr_seq="0.0005 0.001 0.002"
bs_seq="2 3 4 5 6"
np_seq="32 40 48 56 64"

jj=0
for lr in $lr_seq; do
  for bs in $bs_seq; do
    for np in $np_seq; do
      jj=$((jj+1))
      echo "##### ITERATION: "$jj" #####"
      echo "learning rate: "$lr", batch-size: "$bs", # params: "$np
      echo "--- Cell type: INFLAMMATORY ---"
      python -u script_mdl_cell.py --cells $cell_inflam --num_epochs $num_epochs --epoch_check $epoch_check --batch_size $bs --learning_rate $lr --num_params $np

      echo "--- Cell type: EOSINOPHIL ---"
      python -u script_mdl_cell.py --cells $cell_eosin --num_epochs $num_epochs --epoch_check $epoch_check --batch_size $bs --learning_rate $lr --num_params $np

      #return
    done
  done
done

echo "end of shell"
