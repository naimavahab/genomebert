#!/bin/sh
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' 

export CUDA_LAUNCH_BLOCKING=1 

export TORCH_USE_CUDA_DSA=1 

nohup composer main.py yamls/main/mosaic-bert-base-uncased.yaml > mosaicbertv2.log 2>&1 &
