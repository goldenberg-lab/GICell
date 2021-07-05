#!/bin/bash

# NOTE: ~GIOrdinal/data/cropped/{train/test} needs to be copied over from RT5362WL for seeding

# nfill impacts cell x ratio
nfill=1
s2=2

echo "--- (1) CHECK ALIGNMENT BETWEEN ORDINAL AND CELL ---"
python check_crop_GI.py
# Print statements: only 6EAWUIY4_Cecum_55 (due to Cecum vs Cecum-001)

echo "--- (2) CHECK CONDA ENVIRONMENT ---"
source set_env.sh

echo "--- (3) CHECKING CUDA/ANNO STATUS ---"
python check_cuda.py
python check_anno_status.py

echo "--- (4) GENERATE DATA ---"
# nfill:    Number of pixels to pad around annotation point
# s2:       Variance of gaussian blur
python process_Xy.py --nfill $nfill --s2 $s2
# output: ~output/di_img_point.pickle
#         contains all images/annotation points?

echo "--- (5) GENERATE SUMMARY STATS ---"
# output will be used for model run
python explore_data.py
# output:   ~/output/figures/{}.png
#           ~/output/train_val_test_ids.csv

echo "--- (6) TEST MODEL ---"
pmax=8
echo "Testing INFLAM"
python -u run_mdl.py --is_inflam --nepoch 1 --p $pmax

echo "Testing EOSIN"
python -u run_mdl.py --is_eosin --nepoch 1 --p $pmax

echo "--- (7) HP SEARCH ---"

# Call pipeline nohup on appropriate machine
python -u explore_hp.py

return

# # Set folders
# dir_output="/mnt/d/projects/GIcell/output"
# dir_snapshot=$dir_output/checkpoint/snapshot
# sq_output="~/Documents/projects/GI/GICell/output"
# sq_checkpoint=$sq_output/checkpoint
# sq_snapshot=$sq_checkpoint/snapshot


# echo "--- (4) HYPERPARAMETER SELECT ---"
# scp erik@172.16.18.177:$sq_output $dir_output
# scp erik@172.16.18.177:$sq_snapshot/* $dir_snapshot

# echo "--- (5) FULL IMAGE INFERENCE ---"
# scp erik@172.16.18.177:$sq_output/df_fullimg.csv $dir_output

# echo "--- (6) EVALUATION ---"
# # (i) Calculate statistical associations with full-image
# # (ii) Compares current to previous model (visual + scatter)
# #python script_inference.py

echo "--------  END OF pipeline_GI.sh ----------"
