FROM python:3.10-slim-bullseye

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

## Philosopher's Stone (NEJM AI 2026) — transfer learning from 36k PSG recordings
RUN git clone https://github.com/bdsp-core/philosophers-stone /ps

## Install PS-specific dependencies (others already in requirements.txt)
RUN pip install h5py psutil

## Pre-download model weights from HuggingFace into image
## (~1 GB, baked in so inference works without internet)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('/ps/model_files', exist_ok=True); \
path = hf_hub_download(\
    repo_id='wolfgang-ganglberger/philosophers-stone', \
    filename='SleepPhilosophersStone.ckpt', \
    local_dir='/ps/model_files', \
    local_dir_use_symlinks=False); \
print(f'PS model ready: {path}') \
" || echo "WARNING: PS model download failed — will retry at runtime"
