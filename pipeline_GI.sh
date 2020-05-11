#!/bin/bash

# Create environment if does not exist
source env_pipeline.sh

echo "---- CHECKING CUDA STATUS ----"
python script_check_cuda.py

# Create data
python script_data_gen.py

# Train model for each cell type

python script_mdl_cell.py --cells eosonphils --num_epochs 500 --batch_size 2 --learning_rate 0.001 --num_params 16
