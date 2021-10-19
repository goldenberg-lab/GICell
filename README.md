This repository contains the scripts needed to reproduce the eosinophil cell counting analysis. Code requires the data to be set up in `GICell` `GIOrdinal` appropriately. 

To process data and generate models across different hyperparameters, first run `0_pipeline_hp.sh`. Next, run the `6_pipeline_inf.sh` to carry out inference. 