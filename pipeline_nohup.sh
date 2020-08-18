#!/bin/bash

conda activate QuPath

log_files="eosin.log inflam.log cell.log"

for file in $log_files; do
	echo $file
	if test -f ../$file; then
		echo "removing "$file
		rm ../$file
	fi
done

cell_inflam="eosinophil,neutrophil,plasma,lymphocyte"
cell_eosin="eosinophil"

python script_mdl_cell.py --cells $cell_inflam --num_epochs 500 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_check 50
python script_mdl_cell.py --cells $cell_eosin --num_epochs 500 --batch_size 2 --learning_rate 0.001 --num_params 32 --epoch_chec 50


#nohup python -u script_mdl_cell.py --cells $cell_inflam --num_epochs 10 --batch_size 1 --learning_rate 0.001 --num_params 32 --epoch_check 5 > ../inflam.log &
#nohup python -u script_mdl_cell.py --cells $cell_eosin --num_epochs 10 --batch_size 1 --learning_rate 0.001 --num_params 32 --epoch_check 5 > ../eosin.log &



