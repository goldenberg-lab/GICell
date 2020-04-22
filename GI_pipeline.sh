#!/bin/bash

# Create environment if does not exist
conda activate QuPath

echo "---- CHECKING CUDA STATUS ----"
python conda_check.py

# Create data
python data_gen.py

# Train model

