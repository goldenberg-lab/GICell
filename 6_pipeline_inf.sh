#!/bin/bash

# Shell script to analyze results from 0_pipeline_hp.sh

echo "--- (7) HP SEARCH ---"
# Call pipeline_best.sh on appropriate machine to get results
python -u 7_explore_hp.py
# output:   ~/output/figures/gg_{metric}_val.png
#           ~/output/figures/dat_{pr/ce}_ce.csv

echo "--- (8) SELECT BEST MODELS ---"
source 8_pipeline_best.sh


echo "--------  END OF 6_pipeline_hp.sh ----------"