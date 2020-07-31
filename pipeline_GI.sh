#!/bin/bash

# Create environment if does not exist
source pipeline_env.sh

echo "---- CHECKING CUDA STATUS ----"
python script_check_cuda.py

# Create data
python script_data_gen.py

# Train model for each cell type
python script_mdl_cell.py --cells eosinophil --num_epochs 1000 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 200
python script_mdl_cell.py --cells eosinophil,neutrophil,plasma,lymphocyte --num_epochs 1000 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 200

#python script_mdl_cell.py --cells eosinophil,neutrophil,plasma,lymphocyte,other,enterocyte --num_epochs 250 --batch_size 2 --learning_rate 0.001 --num_params 16 --epoch_check 50

# Create the figures to assess performance
python script_eval.py

# Examine best-performing models
python script_inference.py


