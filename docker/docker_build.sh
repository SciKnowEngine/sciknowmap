#!/bin/bash
USER_NAME=$(whoami)
USER_ID=$(id -u $USER_NAME)
echo "Creating sciknowmap system for $USER_NAME $USER_ID"
echo docker build --build-arg USER_ID=$USER_ID --build-arg USER_NAME=$USER_NAME -t sciknowmap .
docker build --build-arg USER_ID=$USER_ID --build-arg USER_NAME=$USER_NAME -t sciknowmap .
echo
echo COMPLETE
