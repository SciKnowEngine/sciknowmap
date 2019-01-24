#!/bin/bash

#
# INVOKE THIS SCRIPT TO RUN A SCIKNOWMAP DOCKER CONTAINER
# - We will store document
#

if [[ $# -ne 2 ]] ; then
  echo "USAGE ./start_docker.sh <DATA>"
  exit
fi

DATA=$2

#  -v option mounts the place where this command was run from as /tmp/evidence_extractor
docker run -i -t --user $(id -u):$(id -g): -p 8888:8888 -p 8889:8889 -p 8787:8787 -v $DATA:/tmp/data/ -v $PWD:/tmp/sciknowmap -w=/tmp/sciknowmap/ --rm sciknowmap
