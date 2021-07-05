#!/bin/bash

dir_code=/home/erik/Documents/projects/GI/GICell/code
cd $dir_code

# Set the conda environment (not conda has not been initiatlized for the nohup bash)
source /home/erik/anaconda3/etc/profile.d/conda.sh
source set_env.sh

# Check CUDA
python -u check_cuda.py

# Loop over different learning rate/batch_size/architecture configurations
nepoch=90
lr_seq="0.001 0.002"
bs_seq="2 4 6"
p_seq="32 64"

jj=0
for lr in $lr_seq; do
  for bs in $bs_seq; do
    for p in $p_seq; do
      jj=$((jj+1))
      echo "##### ITERATION: "$jj" #####"
      echo "learning rate: "$lr", batch-size: "$bs", # params: "$p
      echo "--- Cell type: INFLAMMATORY ---"
      python -u run_mdl.py --is_inflam --nepoch $nepoch --batch $bs --lr $lr --p $p

      echo "--- Cell type: EOSINOPHIL ---"
      python -u run_mdl.py --is_eosin --nepoch $nepoch --batch $bs --lr $lr --p $p
    done
  done
done

echo "end of shell"
