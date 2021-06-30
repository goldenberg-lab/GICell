#!/bin/bash

# scripts to be run on GPU machines without hangup

dir_code=/home/erik/Documents/projects/GI/GICell/code
cd $dir_code

echo " ------ (1) UNet Hyperparameters ------ "
dir_nohup=/data/GICell/output/nohup
log_file=$(date | awk '{gsub(/ /,"_")}1')
log_file=$log_file".log"

nohup ./pipeline_unet_hp.sh > $dir_nohup/$log_file 2>&1 &


echo "end of pipeline_nohup.sh"
