#!/bin/bash

nfill=1  # nfill impacts cell x ratio
s2=2  # Standard deviation of gaussian smoothing
annotators="oscar dua"  # Which folders are inter-annotators
pval=0.2  # What percent of the training data
ds_test="oscar dua 70602"  # Which folders are for testing

echo "--- CHECK CONDA ENVIRONMENT ---"
source set_env.sh

echo "--- (1.A) CHECK ALIGNMENT BETWEEN ORDINAL AND CELL ---"
python -u 1a_check_crop.py
# Print statements: only 6EAWUIY4_Cecum_55 (due to Cecum vs Cecum-001)

echo "--- (1.B) CHECKING CUDA ---"
# Check whether CUDA-tensor is possible
python -u 1b_check_cuda.py

echo "--- (1.C) CHECKING ANNO STATUS ---"
python -u 1c_check_anno.py

echo "--- (1.D) IMAGE TRANS ---"
# Confirm that image rotations/flips work as expected
python -u 1d_check_trans.py

echo "--- (2) GENERATE DATA ---"
# nfill:    Number of pixels to pad around annotation point
# s2:       Variance of gaussian blur
python -u 2_process_xy.py --nfill $nfill --s2 $s2
# output:   ~/output/{df_cells.csv, df_pts.csv} (i)
#           ~/output/annot_{cinci,hsk}.pickle   (ii)
# (i) location of the cell positions and total cell count by ID
# (ii) dictionary with image and labels (with Gaussian blur)

echo "--- (3) SPLIT DATA INTO TRAIN/VAL/DATA ---"
# pval:     Percent of non-test folders to apply random split to
# ds_test:  Folders that should be reserved for testing
python -u 3_data_split.py --pval $pval --ds_test $ds_test
# output:   ~/output/train_val_test.csv
# output:   ~/output/figures/labels/{cell}_{share/n}.png

echo "--- (4) TEST MODEL ---"
# without the --check_model flag
# output:   ~/output/figures/gg_count_pct_{cell}.png
#           ~/output/checkpoint/{cell}/HASH.pkl == {'hp', 'ce_auc', 'pr', 'mdl'*}
#           *requires --save_model flag
echo "Testing INFLAM"
python -u 4_run_mdl.py --is_inflam --check_model
echo "Testing EOSIN"
python -u 4_run_mdl.py --is_eosin  --check_model

# echo "--- (5) RUN OVER HYPERPARAMETERS ---"
# source pipeline_nohup.sh

# echo "--- (6) HP SEARCH ---"
# # Call pipeline_best.sh on appropriate machine to get results
# python -u explore_hp.py
# # output:   ~/output/figures/gg_{metric}_val.png
# #           ~/output/figures/dat_{pr/ce}_ce.csv



# echo "--- (10) Find peak eosin region ---"
# # Find regions of highest eosinophil density on full image
# python -u find_peak_eosin.py --nfill 1 --hw 500 --stride 500 --hsk --cinci


# echo "--- (7) Test set ---"
# # Find test-set performance
# python -u explore_test.py
# # output:   ~/output/inf_stab.csv
# #           ~/output/figures/{gg_inf_stab,gg_auroc_tt,gg_auprc,gg_scatter_unet,gg_thresh_n_{msr},gg_scatter_star,gg_perf_star}

# echo "--- (8) Inter-annotator variability ---"
# python -u explore_inter.py --annotators $annotators



echo "--------  END OF pipeline_GI.sh ----------"