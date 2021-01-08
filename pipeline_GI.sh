#!/bin/bash

# Create environment if does not exist
#source pipeline_env.sh

echo "---- CHECKING CUDA STATUS ----"
python script_check_cuda.py

# ------ (1) Create data ------ #
python script_data_gen.py

root=$(pwd | cut -d"/" -f2)
if [ $root == "mnt" ]; then
  dir_output="/mnt/d/projects/GIcell/output"
  dir_snapshot=$dir_output/checkpoint/snapshot
  # ------ (3) Hyperparameter select ------ #
  sq_output="~/Documents/projects/GI/GICell/output"
  sq_checkpoint=$sq_output/checkpoint
  sq_snapshot=$sq_checkpoint/snapshot
  scp erik@172.16.18.177:$sq_output $dir_output
  scp erik@172.16.18.177:$sq_snapshot/* $dir_snapshot

  # ------ (4) Full image inference ------ #
  # Load the merged slices
  scp erik@172.16.18.177:$sq_output/df_fullimg.csv $dir_output

else

  # ------ (2) Train models ------ #

  nohup sh pipeline_nohup.sh > ../cell.log 2>&1 &

  # ------ (3) Hyperparameter select ------ #

  python script_hyperparameter.py  # ~/output/df_hp_perf.csv
  python script_tensorboard.py  # Creates figures

  # ------ (4) Full image inference ------ #

  # Run on GPU with at least 11G
  for ii in {0..188..1}; do
    echo "Image: "$ii
    python script_fullimg.py --ridx $ii --kk 2500
  done

  # Merge the slices
  python script_merge_slices.py

  # ------ (5) Evaluation ------ #

  # (i) Calculate statistical associations with full-image
  # (ii) Compares current to previous model (visual + scatter)

  #python script_inference.py

fi

echo "--------  END OF pipeline_GI.sh ----------"
