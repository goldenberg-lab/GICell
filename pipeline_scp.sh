#!/bin/bash

# nohup sh pipeline_nohup.sh > ../cell.log 2>&1 &

# SCRIPT TO ZIP MOST RECENT MODEL RESULTS
dir_check=../output/checkpoint
# Remove existing zip files
files_check=$(ls $dir_check | grep .zip)
nzip=$(($(echo -n $files_check | wc -m)))

if [ $nzip -gt 0 ]; then
	echo "remove existing zip files"
for file in $files_check; do
	echo $file
	rm $dir_check/$file
done
fi

fold_eosin=$dir_check/eosinophil/
fold_inflam=$dir_check/eosinophil_lymphocyte_neutrophil_plasma
date_eosin=$(ls $fold_eosin | grep "^[0-9]\{4\}" | sort -r | head -1)
date_inflam=$(ls $fold_eosin | grep "^[0-9]\{4\}" | sort -r | head -1)

echo "most recent eosin/inflam folder date is: "$date_eosin", "$date_inflam

if [ "$date_inflam" == "$date_eosin" ]; then
	echo "folder dates line up"
	zip -r $dir_check/eosin.zip $fold_eosin/$date_eosin
	zip -r $dir_check/inflam.zip $fold_inflam/$date_inflam
else
	echo "inflam/eosin do not have the same dates"
fi

