This repository contains the scripts needed to reproduce the eosinophil cell counting analysis. Code requires the data to be set up in `GICell` `GIOrdinal` appropriately. 

To process data and generate models across different hyperparameters, please run the shells scripts in the following order:

1. `0_pipeline_hp.sh`: Processes data and train models across different hyperparameter configurations.
2. `6_pipeline_inf.sh`: Finds optimal hyperparameters and retrains models with these settings. 
3. `9_analyze_test.sh`: Uses "best" models to carry out inference on test sets.