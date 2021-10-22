#!/bin/bash

dir_code=$1
path_conda=$2
dir_output=$3
echo "dir_code = "$dir_code
echo "path_conda = "$path_conda

cd $dir_code

# Set the conda environment (not conda has not been initiatlized for the nohup bash)
source $path_conda
source set_env.sh

# Check CUDA
python -u 1b_check_cuda.py

# Get the optimal hyperparameters
path_hp=$dir_output/hp_best.csv
if [[ -f "$path_hp" ]]; then
    echo "file exists"
    cn_ord=$(cat $path_hp | head -1 |awk '{gsub(","," ",$0); print}')
    for ii in 1 2; do
        echo $ii
        tmp_row=$(cat $path_hp | tail -2 | head -$ii | tail -1)
        val_ord=$(echo $tmp_row | awk '{gsub(","," ",$0); print}')
        jj=0
        for val in $val_ord; do
            jj=$(($jj + 1))
            rx='{split($0,a," "); print a['$jj']}'
            cn_jj=$(echo $cn_ord | awk "$rx")
            echo "jj="$jj", cn="$cn_jj", val="$val
            eval $cn_jj"="$val
        done
        if [[ "$cell" == "inflam" ]]; then
            echo "Training inflammatory model"
            python -u 4_run_mdl.py --is_inflam --nepoch $epoch --batch $batch --lr $lr --p $p --save_model
        else
            echo "Training eosinophil model"
            python -u 4_run_mdl.py --is_eosin --nepoch $epoch --batch $batch --lr $lr --p $p --save_model
        fi
    done
else
    echo "file does not exist"
    return
fi


echo "~~ End of pipeline_unet_best.sh ~~~"