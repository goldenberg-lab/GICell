#!/bin/bash

# Will copy files from ~/output/checkout/{cell}/* to different machinse

cpu=$(hostname)

remote_cavansite=erik@cavansite.sickkids.ca
remote_snowqueen=erik@snowqueen.sickkids.ca
dir_snowqueen=/data/GICell/output/checkpoint/
dir_cavansite=/data/erik/GICell/output/checkpoint/

if [[ $cpu == "snowqueen" ]]; then
    echo "copying from cavansite to snowqueen"
    scp $remote_cavansite:$dir_cavansite/eosin/*.pkl $dir_snowqueen/eosin/
    scp $remote_cavansite:$dir_cavansite/inflam/*.pkl $dir_snowqueen/inflam/
fi
