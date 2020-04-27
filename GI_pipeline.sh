#!/bin/bash

# Create environment if does not exist
source env_pipeline.sh

echo "---- CHECKING CUDA STATUS ----"
python conda_check.py

# Create data
#python data_gen.py

# Train model
#python train_unet.py
