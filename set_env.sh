#!/bin/bash

# Check to see if miniconda environment exists
grep_env=$(ls ~/miniconda3/envs | grep QuPath)
n_char=$(echo $grep_env | wc -w)

if [[ "$n_char" -eq 0 ]]; then
    echo "Installing environment"
    conda create --name QuPath --file conda_env.txt python=3.7
else
    echo "Environment already exists"
fi

conda activate QuPath

# Check to see if hickle is installed
grep_env=$(conda env export | grep hickle)
n_char=$(echo $grep_env | wc -w)

if [[ "$n_char" -eq 0 ]]; then
    echo "Installing hickle"
    python3 -m pip install hickle==4.0.4
else
    echo "hickle already exists"
fi

conda list --explicit > conda_env.txt  # git to detect if env has changed


