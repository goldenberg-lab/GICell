#!/bin/bash

# Analyze results using the optimally-trained models

nfill=1  # nfill impacts cell x ratio
annotators="oscar dua"  # Which folders are inter-annotators
# ds_test="oscar dua 70608"  # Which folders are for testing

echo "--- (10) RENAME OPTIMAL MODELS ---"
python -u 10_rename_mdls.py
#       ~/output/{hash.pickle -> best_cell.pickle}

echo "--- (11) EXPLORE TEST SET ---"
python -u 11_explore_test.py --nfill $nfill --check_flips
#       ~/output/figures/{*.png}/
#       ~/output/figures/inference/{*.png}

# echo "--- (12) EXPLORE INTER-ANNOTATOR ---"
python -u 12_explore_inter.py --nfill $nfill --annotators $annotators


echo "~~~ End of 9_analyze_test.sh ~~~"