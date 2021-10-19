#!/bin/bash

# Shell script to analyze results from 0_pipeline_hp.sh

echo "--- (7) HP SEARCH ---"
# Call pipeline_best.sh on appropriate machine to get results
python -u 7_explore_hp.py
# output:   ~/output/figures/gg_{metric}_val.png
#           ~/output/figures/dat_{pr/ce}_ce.csv

# echo "--- (8) SELECT BEST MODELS ---"
# source 8_pipeline_best.sh

# echo "--- (9) Test set ---"
# # Find test-set performance
# python -u 9_explore_test.py
# # output:   ~/output/inf_stab.csv
# #           ~/output/figures/{gg_inf_stab,gg_auroc_tt,gg_auprc,gg_scatter_unet,gg_thresh_n_{msr},gg_scatter_star,gg_perf_star}

# echo "--- (10) Inter-annotator variability ---"
# python -u explore_inter.py --annotators $annotators


# echo "--- (11) Find peak eosin region ---"
# # Find regions of highest eosinophil density on full image
# python -u find_peak_eosin.py --nfill 1 --hw 500 --stride 500 --hsk --cinci





echo "--------  END OF 0_pipeline_hp.sh ----------"