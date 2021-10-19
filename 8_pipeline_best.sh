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

echo "--- Cell type: INFLAMMATORY ---"
nepoch_inflam=85
batch_inflam=4
lr_inflam=0.003
p_inflam=96

python -u run_mdl.py --is_inflam --nepoch $nepoch_inflam --batch $batch_inflam --lr $lr_inflam --p $p_inflam --save_model

echo "--- Cell type: EOSINOPHIL ---"
nepoch_eosin=56
batch_eosin=4
lr_eosin=0.002
p_eosin=64

python -u run_mdl.py --is_eosin --nepoch $nepoch_eosin --batch $batch_eosin --lr $lr_eosin --p $p_eosin --save_model

echo "end of shell"

