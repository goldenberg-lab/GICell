#!/bin/bash

# scripts to be run on GPU machines

# ------ (2) Train models ------ #

nohup sh pipeline_nohup.sh > ../cell.log 2>&1 &

# ------ (3) Hyperparameter select ------ #

python script_hyperparameter.py  # ~/output/df_hp_perf.csv
python script_tensorboard.py  # Creates figures

# ------ (4) Full image inference ------ #

# Run on GPU with at least 11G
for ii in {0..188..1}; do
  echo "Image: "$ii
  python script_fullimg.py --ridx $ii --kk 2500
done

# Merge the slices
python script_merge_slices.py
