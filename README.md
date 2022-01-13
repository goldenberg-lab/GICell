This repository contains the scripts needed to reproduce the eosinophil cell counting analysis. The code requires assumes that the folder structure is set up as follows:

├── GICell
│   ├── code
│   ├── images
│   ├── output
│   └── points
└── GIOrdinal
    └── data

There are three main shell scripts which need to be run in a specific order (shown below). A description of each file in the script is summarized below. Please review the shell scripts themselves as they provide additional commentary on the python scripts. 

1. `0_pipeline_hp.sh`: Processes data and trains models across different hyperparameter configurations. Wait for `5_pipeline_nohup.sh` to finish running before moving onto the next step. 
    1. `set_env.sh`: This will look to see if your conda installation has an environment called QuPath. If the environment is not found, it will be created with the necessary packages found in `conda_env.txt`. Note that if you do not have miniconda installed, you may need to change line 4.
    2. `1a_check_crop.py`: This is a sanity check script to make sure that there are labels found for the different crops. Change the `find_dir_cell` function to the path the directory structure shown above.
    3. `1b_check_cuda.py`: Makes sure that CUDA is configured properly and that PyTorch can connect to a GPU device.
    4. `1c_check_anno.py`: Provides a print read-out of the different annotations and which folders they live in (as well if there are any images that do not yet have annotations). 
    5. `1d_check_trans.py`: Makes sure that the image encoder can apply loss-less rotations and flips (up to some floating point errors).
    6. `2_process_xy.py`: Loads all annotations and crops and saves an pickled dictionary of data for each annotator.
    7. `3_data_split.py`: Assigns images into training, validation, and test.
    8. `4_run_mdl.py`: Checks that model class works with --check_model argument (will run for one epoch and produce no output)
    9. `5_pipeline_nohup.sh`: Runs `pipeline_unet_hp.sh` with no hang up, which will call `4_run_mdl.py` with different hyperparameter configurations for learning rate, batch size, and model complexity. The dir_code, path_conda, and dir_nohup variables will need to be changed for a new machine.
2. `6_pipeline_inf.sh`: Determines which hyperparameters are optimal and retrains models with these settings. Wait for `8_pipeline_best.sh` to finish running before moving onto the next step.  
    1. `7_explore_hp.py`: Determines what are the "best" hyperparameters in terms of AUROC
    2. `8_pipeline_best.sh`: Calls in `pipeline_unet_best.sh` with no hang up, and uses the output from the previous script to set hyperparameters for exact training schedule.
3. `9_analyze_test.sh`: Uses "best" models to carry out inference on test sets.
    1. `10_rename_mdls.py`: Changes the hashed model name to best_eosin.pickle.
    2. `11_explore_test.py`: Gets aggregate performance as well as individualized inferences on different datasets using the best model.
    3. `12_explore_inter.py`: Compares inter-annotator variability of pathologists, to that of the model and pathologists.