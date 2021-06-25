#!/bin/bash

source set_env.sh

# Loop over different learning rate/batch_size/architecture configurations
nepoch=90
epoch_check=15
lr_seq="0.0005 0.001 0.002"
bs_seq="2 3 4 5 6"
p_seq="32 40 48 56 64"

jj=0
for lr in $lr_seq; do
  for bs in $bs_seq; do
    for p in $p_seq; do
      jj=$((jj+1))
      echo "##### ITERATION: "$jj" #####"
      echo "learning rate: "$lr", batch-size: "$bs", # params: "$np
      echo "--- Cell type: INFLAMMATORY ---"
      python -u script_mdl_cell.py --is_inflam --nepoch $nepoch --batch $bs --lr $lr --p $p

      echo "--- Cell type: EOSINOPHIL ---"
      python -u script_mdl_cell.py --is_eosin --nepoch $nepoch --batch $bs --lr $lr --p $p

      #return
    done
  done
done

echo "end of shell"
