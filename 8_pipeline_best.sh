#!/bin/bash

# scripts to be run on GPU machines without hangup
cpu=$(hostname)
if [[ $cpu == "snowqueen" ]]; then
    echo "on snowqueen"
    dir_code=/home/erik/Documents/projects/GI/GICell/code
    path_conda=/home/erik/miniconda3/etc/profile.d/conda.sh
    dir_output=/data/GICell/output
    dir_nohup=$dir_output/nohup
elif [[ $cpu == "cavansite" ]]; then
    echo "on cavansite"
    dir_code=/home/erik/Documents/projects/GICell
    path_conda=/opt/programs/anaconda3/etc/profile.d/conda.sh
    dir_output=/data/erik/GICell/output
    dir_nohup=$dir_output/nohup
elif [[ $cpu == "malachite" ]]; then
    dir_code=/home/erik/projects/GICell/code
    path_conda=/home/erik/miniconda3/etc/profile.d/conda.sh
    dir_output=/home/erik/projects/GICell/output
    dir_nohup=$dir_output/nohup
else
    echo "somewhere else"
fi

log_file=$(date | awk '{gsub(/ /,"_")}1')
log_file=$log_file".log"

nohup ./pipeline_unet_best.sh $dir_code $path_conda $dir_output > $dir_nohup/$log_file 2>&1 &

echo "~~~ End of 8_pipeline_best.sh ~~~"