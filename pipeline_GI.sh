#!/bin/bash

# Create environment if does not exist
#source pipeline_env.sh

echo "---- CHECKING CUDA STATUS ----"
python script_check_cuda.py

#################################
# ------ (1) Create data ------ #
python script_data_gen.py


##################################
# ------ (2) Train models ------ #

nohup sh pipeline_nohup.sh > ../cell.log 2>&1 &
#python script_mdl_cell.py --cells eosinophil --num_epochs 500 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 50

###########################################
# ------ (3) Hyperparameter select ------ #

python script_hyperparameter.py  # ~/output/df_hp_perf.csv

root=$(pwd | cut -d"/" -f2)
if [ $root == "mnt" ]; then
  dir_output="/mnt/d/projects/GIcell/output"
  dir_snapshot=$dir_output/checkpoint/snapshot
  scp erik@172.16.18.177:~/Documents/projects/GIProject/cell_counter/output/df_hp_perf.csv $dir_output
  scp erik@172.16.18.177:~/Documents/projects/GIProject/cell_counter/output/checkpoint/snapshot/* $dir_snapshot
fi

source pipeline_scp.sh  # Run on predator after running above on snowqueen
python script_tensorboard.py  # Creates figures

##########################################
# ------ (4) Full image inference ------ #

# Run on GPU with at least 11G
for ii in {0..189..1}; do
  echo "Image: "$ii
  python script_fullimg.py --ridx $ii
done

#if [ $root == "mnt" ]; then
#  scp erik@172.16.18.177:~/Documents/projects/GIProject/cell_counter/output/figures/
#fi


################################
# ------ (5) Evaluation ------ #

# (i) Calculate statistical associations with full-image
# (ii) Compares current to previous model (visual + scatter)

python script_inference.py

echo "--------  END OF pipeline_GI.sh ----------"