#!/bin/bash

# Create environment if does not exist
#source pipeline_env.sh

echo "---- CHECKING CUDA STATUS ----"
python script_check_cuda.py

# Create data
#python script_data_gen.py

# Train model for each cell type
#python script_mdl_cell.py --cells eosinophil --num_epochs 500 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 50
#python script_mdl_cell.py --cells eosinophil,neutrophil,plasma,lymphocyte --num_epochs 1000 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 200

# Create the figures to assess performance
#python script_eval.py

# Generate the patient data
for ii in {0..188..1}; do
  echo $ii
  python script_fullimg.py --ridx $ii
done

# Examine best-performing models
python script_inference.py



