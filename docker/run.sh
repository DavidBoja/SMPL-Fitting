#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "You need to supply CODE and (optionally) DATA paths as specified in the README"
    exit 1
fi

CODE_DIR_PATH=$1

if [ -z "$2" ]
  then
    # run docker image without data path mounting
    docker run --gpus all --detach --shm-size=8gb --name smpl-fitting-container -t -v $CODE_DIR_PATH/:/SMPL-Fitting smpl-fitting
    exit 1
fi

DATA_DIR_PATH=$2
docker run --gpus all --detach --shm-size=8gb --name smpl-fitting-container -t -v $CODE_DIR_PATH/:/SMPL-Fitting -v $DATA_DIR_PATH/:/data smpl-fitting
