FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

## Philosopher's Stone (NEJM AI 2026) — transfer learning from 36k PSG recordings
## Use archived repos (Debian Buster is EOL)
RUN echo "deb http://archive.debian.org/debian buster main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main" >> /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*
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