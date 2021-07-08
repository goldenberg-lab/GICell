#!/bin/bash

dir_code=$1
path_conda=$2
echo "dir_code = "$dir_code
echo "path_conda = "$path_conda

cd $dir_code

# Set the conda environment (not conda has not been initiatlized for the nohup bash)
source $path_conda
source set_env.sh

# Check CUDA
python -u check_cuda.py

# Loop over different learning rate/batch_size/architecture configurations
nepoch=90
lr_seq="0.002"
bs_seq="6"
p_seq="64"

jj=0
for lr in $lr_seq; do
  for bs in $bs_seq; do
    for p in $p_seq; do
      jj=$((jj+1))
      echo "##### ITERATION: "$jj" #####"
      echo "learning rate: "$lr", batch-size: "$bs", # params: "$p
      echo "--- Cell type: INFLAMMATORY ---"
      python -u run_mdl.py --is_inflam --nepoch $nepoch --batch $bs --lr $lr --p $p --save_model

      echo "--- Cell type: EOSINOPHIL ---"
      python -u run_mdl.py --is_eosin --nepoch $nepoch --batch $bs --lr $lr --p $p --save_model
    done
  done
done

echo "end of shell"
