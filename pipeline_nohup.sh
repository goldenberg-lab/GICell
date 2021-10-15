#!/bin/bash

# scripts to be run on GPU machines without hangup
cpu=$(hostname)
if [[ $cpu == "snowqueen" ]]; then
    echo "on snowqueen"
    dir_code=/home/erik/Documents/projects/GI/GICell/code
    path_conda=/home/erik/miniconda3/etc/profile.d/conda.sh
    dir_nohup=/data/GICell/output/nohup
elif [[ $cpu == "cavansite" ]]; then
    echo "on cavansite"
    dir_code=/home/erik/Documents/projects/GICell
    path_conda=/opt/programs/anaconda3/etc/profile.d/conda.sh
    dir_nohup=/data/erik/GICell/output/nohup
elif [[ $cpu == "malachite" ]]; then
    dir_code=/home/erik/projects/GICell/code
    path_conda=/home/erik/miniconda3/etc/profile.d/conda.sh
    dir_nohup=/home/erik/projects/GICell/output/nohup
else
    echo "somewhere else"
fi

cd $dir_code

echo " ------ (1) UNet Hyperparameters ------ "

log_file=$(date | awk '{gsub(/ /,"_")}1')
log_file=$log_file".log"

nohup ./pipeline_unet_hp.sh $dir_code $path_conda > $dir_nohup/$log_file 2>&1 &
# nohup ./pipeline_best.sh $dir_code $path_conda > $dir_nohup/$log_file 2>&1 &

echo "end of pipeline_nohup.sh"
