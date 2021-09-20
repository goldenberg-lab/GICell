#!/bin/bash

# Check to see if miniconda environment exists
grep_env=$(ls ~/miniconda3/envs | grep QuPath)
n_char=$(echo $grep_env | wc -w)

if [[ "$n_char" -eq 0 ]]; then
    echo "Installing environment"
    conda create --name QuPath --file env_qupath.txt python=3.9
else
    echo "Environment already exists"
fi

conda activate QuPath
