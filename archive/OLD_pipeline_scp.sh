#!/bin/bash

root=$(pwd | cut -d"/" -f2)
if [ $root == "mnt" ]; then
  echo "Script is running from predator"
  # Set the baseline director
  dir_base="/mnt/d/projects/GIcell"
  dir_output=$dir_base"/output"
  dir_snapshot=$dir_output/checkpoint/snapshot
  echo "copying df_hp_perf snowqueen to local"
  scp erik@172.16.18.177:~/Documents/projects/GIProject/cell_counter/output/df_hp_perf.csv $dir_output
  echo "copying snapshot/* snowqueen to local"
  scp erik@172.16.18.177:~/Documents/projects/GIProject/cell_counter/output/checkpoint/snapshot/* $dir_snapshot
fi


## SCRIPT TO ZIP MOST RECENT MODEL RESULTS
#dir_check=../output/checkpoint
#root=$(pwd | cut -d"/" -f2)
#echo "root "$root
#
## Remove existing zip files
#files_check=$(ls $dir_check | grep .zip)
#nzip=$(($(echo -n $files_check | wc -m)))
#
#if [ $nzip -gt 0 ]; then
#  echo "remove existing zip files"
#for file in $files_check; do
#  echo $file
#  rm $dir_check/$file
#done
#fi
#
#cells="eosinophil eosinophil_lymphocyte_neutrophil_plasma"
#
#if [ $root == "mnt" ]; then
#  echo "Script is running from local"
#  scp erik@snowqueen.sickkids.ca:/home/erik/Documents/projects/GIProject/cell_counter/output/checkpoint/\{eosin.zip,inflam.zip\} $dir_check/.
#  for cell in $cells; do
#    if [ $cell == 'eosinophil' ]; then
#      cell2="eosin.zip"
#    else
#      cell2="inflam.zip"
#    fi
#    echo "cell: "$cell", cell2: "$cell2
#    mkdir $dir_check/tmp
#    unzip $dir_check/$cell2 -d $dir_check/tmp
#    cp -r $dir_check/tmp/output/checkpoint/$cell/* $dir_check/$cell
#    rm -r $dir_check/tmp
#  done
#
#else
#  echo "Script is running from remote"
#  fold_eosin=$dir_check/eosinophil/
#  fold_inflam=$dir_check/eosinophil_lymphocyte_neutrophil_plasma
#  date_eosin=$(ls $fold_eosin | grep "^[0-9]\{4\}" | sort -r | head -1)
#  date_inflam=$(ls $fold_eosin | grep "^[0-9]\{4\}" | sort -r | head -1)
#
#  echo "most recent eosin/inflam folder date is: "$date_eosin", "$date_inflam
#
#  if [ "$date_inflam" == "$date_eosin" ]; then
#    echo "folder dates line up"
#    zip -r $dir_check/eosin.zip $fold_eosin/$date_eosin
#    zip -r $dir_check/inflam.zip $fold_inflam/$date_inflam
#  else
#    echo "inflam/eosin do not have the same dates"
#  fi
#fi
