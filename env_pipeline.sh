#!/bin/bash

echo "------------------------------"
echo "CREATING QuPath ENVIRONMENT"
echo "------------------------------"

n_env="QuPath"

# Check if conda exists (if not assume on HPF)
#n_conda=$(($(command -v "conda -v" | wc -l)))
n_conda=$(($(command -v "conda" | wc -l)))
echo $n_conda
if [ $n_conda == 1 ]; then
  echo "Conda exists, running locally"
  e_los=$(conda env list | grep $n_env)
  n_los=$(($(echo $e_los | wc -w)))
  if [ $n_los -gt 0 ]; then
    echo $n_env" environment already found"
    conda activate $n_env
  else
    echo $n_env" environment needs to be installed"
    conda create --name $n_env python=3.7
    conda activate $n_env
  fi
else
  echo "Conda does not exist, loading python module"
  module load python/3.7.1
fi


echo "------------------- PYTHON LOCATION -------------------- "
which python

fn="qupath_env.txt" # filename with the different pip environments

# --- install/update packages --- #
n_line=$(cat $fn | grep -n pip | tail -1 | cut -d ":" -f1)
n_line=$(($n_line + 1))
n_end=$(cat $fn | grep -n prefix | tail -1 | cut -d ":" -f1)
n_end=$(($n_end - 1))
echo "line_start:"$n_line
echo "line_end:"$n_end
holder=""
for ii in `seq $n_line $n_end`; do
	echo "line: "$ii
	pckg=$(cat $fn | head -$ii | tail -1 | sed -e "s/[[:space:]]//g")
	pckg=$(echo $pckg | sed -e "s/^\-//g")
	holder=$holder" "$pckg
done
echo "packages: "$holder

if [ $n_conda == 1 ]; then
  echo "Installing for conda environment"
  pip install $holder
else
  printf "\n------------ Installing for HPF ------------\n"
  pip install $holder --user
fi

echo "end of script"
return

