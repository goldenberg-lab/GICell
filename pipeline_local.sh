#!/bin/bash

# <--- Script to run locally to analyze results ---> #

# Set folders
dir_output="/mnt/d/projects/GIcell/output"
dir_snapshot=$dir_output/checkpoint/snapshot
sq_output="~/Documents/projects/GI/GICell/output"
sq_checkpoint=$sq_output/checkpoint
sq_snapshot=$sq_checkpoint/snapshot

echo "--- (0) CHECK ALIGNMENT BETWEEN ORDINAL AND CELL ---"
python check_crop_GI.py

echo "--- (1) CHECK CONDA ENVIRONMENT ---"
source set_env.sh

echo "--- (2) CHECKING CUDA/ANNO STATUS ---"
python check_cuda.py
python check_anno_status.py

echo "--- (3) GENERATE DATA ---"
# nfill:    Number of pixels to pad around annotation point
# s2:       Variance of gaussian blur
python process_Xy.py --nfill 1 --s2 2
# output: ~output/di_img_point.pickle
#         contains all images/annotation points?
return


echo "--- (4) HYPERPARAMETER SELECT ---"
scp erik@172.16.18.177:$sq_output $dir_output
scp erik@172.16.18.177:$sq_snapshot/* $dir_snapshot

echo "--- (5) FULL IMAGE INFERENCE ---"
scp erik@172.16.18.177:$sq_output/df_fullimg.csv $dir_output

echo "--- (6) EVALUATION ---"
# (i) Calculate statistical associations with full-image
# (ii) Compares current to previous model (visual + scatter)
#python script_inference.py

echo "--------  END OF pipeline_GI.sh ----------"
