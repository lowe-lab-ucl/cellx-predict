#!/bin/bash

PATCH_DIR=/media/quantumjot/DataIII/Data/VAE_new/training/patches
GLIMPSE_DIR=/media/quantumjot/DataIII/Data/VAE_new/training/glimpses
MODEL_DIR=/media/quantumjot/DataIII/Models/docker
LOG_DIR=/media/quantumjot/DataIII/Training/cellx-logs


# docker run -u $(id -u):$(id -g) -it --gpus all \
#   -v $PATCH_DIR:/cellx-predict/container/data:ro \
#   -v $LOG_DIR:/cellx-predict/container/logs:rw \
#   -v $MODEL_DIR:/cellx-predict/container/models:rw \
#   --rm cellxpred/cellxpred:latest \
#   python run_training.py \
#     --model=encoder \
#     --epochs=100

# docker run -u $(id -u):$(id -g) -it --gpus all \
#   -v $PATCH_DIR:/cellx-predict/container/data:ro \
#   -v $LOG_DIR:/cellx-predict/container/logs:rw \
#   -v $MODEL_DIR:/cellx-predict/container/models:rw \
#   --rm cellxpred/cellxpred:latest \
#   python run_training.py \
#     --model=projector \
#     --epochs=100

docker run -u $(id -u):$(id -g) -it --gpus all \
  -v $GLIMPSE_DIR:/cellx-predict/container/data:rw \
  -v $LOG_DIR:/cellx-predict/container/logs:rw \
  -v $MODEL_DIR:/cellx-predict/container/models:rw \
  --rm cellxpred/cellxpred:latest \
  python run_prepare.py \
    --model=temporal \
    --use_rotations

# docker run -u $(id -u):$(id -g) -it --gpus all \
#   -v $GLIMPSE_DIR:/cellx-predict/container/data:ro \
#   -v $LOG_DIR:/cellx-predict/container/logs:rw \
#   -v $MODEL_DIR:/cellx-predict/container/models:rw \
#   --rm cellxpred/cellxpred:latest \
#   python run_training.py \
#     --model=temporal \
#     --epochs=100 \
#     --use_probabilistic_encoder
