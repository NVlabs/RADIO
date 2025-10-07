#!/bin/bash

PARTITION=${1:-"interactive"}
GPUS=${2:-2}
PROJECT=${3:-"."}
# GPUS=${2:-8}

# On aws, the correct drives are auto-mounted
MOUNT_CMD=""
IMAGE_SUFFIX=""
DURATION="4"
CONSTRAINT=""
# if [[ $HOSTNAME != "draco-aws-login-01" ]]; then
#     MOUNT_CMD="--mounts $MOUNTS"
#     CONSTRAINT="--constraint 'gpu_32gb'"
# else
#     IMAGE_SUFFIX="-aws"
#     DURATION="4"
# fi

IMAGE="/lustre/fsw/portfolios/llmservice/users/mranzinger/sqsh_images/dler+evfm+radio+latest.sqsh"

WORKDIR=`pwd`
if [[ $WORKDIR != "." ]]; then
    WORKDIR="${WORKDIR}/${PROJECT}"
fi

if [[ $HOSTNAME == "draco-oci-login" ]]; then
    ADDL="--more_srun_args=--gpus-per-node=$GPUS"
fi

submit_job \
           --gpu $GPUS \
           --partition "$PARTITION" \
           $CONSTRAINT \
           $MOUNT_CMD \
           --workdir $WORKDIR \
           --image $IMAGE \
           --coolname \
           --interactive \
           --duration $DURATION \
           $ADDL \
           -c "bash"
