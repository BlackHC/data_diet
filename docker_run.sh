#!/bin/bash
# set common-sense Bash options
set -o errexit -o pipefail -o noclobber -o nounset

# Docker: run image based on Dockerfile
docker build -t data_diet_docker docker/
docker run -v /home/blackhc/PycharmProjects/data_diet:/data_diet \
      --network host -P --runtime=nvidia --ipc host --ulimit stack=67108864 --gpus all \
      data_diet_docker  "$@"
