#!/bin/bash

# NOTE: ~GIOrdinal/data/cropped/{train/test} needs to be copied over from RT5362WL for seeding

# nfill impacts cell x ratio
nfill=1
s2=2
annotators="oscar dua"

echo "--- (1) CHECK ALIGNMENT BETWEEN ORDINAL AND CELL ---"
python -u check_crop_GI.py
# Print statements: only 6EAWUIY4_Cecum_55 (due to Cecum vs Cecum-001)


echo "--- (2) CHECK CONDA ENVIRONMENT ---"
source set_env.sh


echo "--- (3) CHECKING CUDA/ANNO STATUS ---"
# Check whether CUDA-tensor is possible
python -u check_cuda.py

# Print the status of the new/old annotations with total count
python -u check_anno_status.py

# Confirm that image rotations/flips work as expected
python -u check_img_trans.py


echo "--- (4) GENERATE DATA ---"
# nfill:    Number of pixels to pad around annotation point
# s2:       Variance of gaussian blur
python -u process_Xy.py --nfill $nfill --s2 $s2
# output:   ~/output/{df_cells.csv, df_pts.csv} (i)
#           ~/output/annot_{cinci,hsk}.pickle   (ii)
# (i) location of the cell positions and total cell count by ID
# (ii) dictionary with image and labels (with Gaussian blur)


echo "--- (5) GENERATE SUMMARY STATS ---"
# Generate train/val/test split and distribution figures
python -u explore_data.py
# output:   ~/output/train_val_test.csv
# output:   ~/output/figures/labels/{cell}_{share/n}.png


echo "--- (6) TEST MODEL ---"
# without the --check_model flag
# output:   ~/output/figures/gg_count_pct_{cell}.png
#           ~/output/checkpoint/{cell}/HASH.pkl == {'hp', 'ce_auc', 'pr', 'mdl'*}
#           *requires --save_model flag
echo "Testing INFLAM"
python -u run_mdl.py --is_inflam --check_model
echo "Testing EOSIN"
python -u run_mdl.py --is_eosin  --check_model


echo "--- (7) HP SEARCH ---"
# Call pipeline_best.sh on appropriate machine to get results
python -u explore_hp.py
# output:   ~/output/figures/gg_{metric}_val.png
#           ~/output/figures/dat_{pr/ce}_ce.csv


echo "--- (8) Test set ---"
# Find test-set performance
python -u explore_test.py
# output:   ~/output/inf_stab.csv
#           ~/output/figures/{gg_inf_stab,gg_auroc_tt,gg_auprc,gg_scatter_unet,gg_thresh_n_{msr},gg_scatter_star,gg_perf_star}

echo "--- (9) Inter-annotator variability ---"
python -u explore_inter.py --annotators $annotators

echo "--- (10) Find peak eosin region ---"
# Find regions of highest eosinophil density on full image
python -u find_peak_eosin.py --nfill 1 --hw 500 --stride 500 --hsk --cinci

echo "--------  END OF pipeline_GI.sh ----------"