#!/bin/bash

# Create environment if does not exist
source env_pipeline.sh

echo "---- CHECKING CUDA STATUS ----"
python check_cuda.py

# Create data
python data_gen.py

# Train model
#python train_unet.py
