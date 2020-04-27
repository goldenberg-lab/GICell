#!/bin/bash

echo "QuPath processing pipeline"

conda activate QuPath

# Generate the pickle dictionary of data
python data_gen.py

# Run the model
python train_unet.py
