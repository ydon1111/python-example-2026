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

## Patch philosopher_utils.py for PyTorch Lightning 2.x compatibility:
##   1) load_from_checkpoint must be called on the class (not an instance)
##   2) use strict=False to allow checkpoint->model architecture mismatch
RUN python -c "\
with open('/ps/phi_utils/philosopher_utils.py', 'r') as f: c = f.read();\
old = '    model = (\n        SleepPhilosopherSpectral(\n            **model_args,\n            dim_final_latent_space=dim_final_latent_space,\n            fs_time=fs_time,\n        ).load_from_checkpoint(str(model_path))\n    )';\
new = '    model = SleepPhilosopherSpectral.load_from_checkpoint(\n        str(model_path),\n        strict=False,\n    )';\
assert old in c, 'patch target not found';\
c = c.replace(old, new);\
open('/ps/phi_utils/philosopher_utils.py', 'w').write(c);\
print('philosopher_utils.py patched OK')\
"

## Patch philosopher.py: free ssqueezepy wavelet arrays (~5.9 GB each) between
## patients to prevent OOM. gc.collect() alone doesn't release memory to OS;
## malloc_trim(0) forces glibc to return freed pages back to the kernel.
RUN python -c "\
with open('/ps/philosopher.py', 'r') as f: c = f.read();\
old = \"        plt.close('all')\n        \n    if torch.cuda.is_available():\";\
new = \"        plt.close('all')\n        del specs\n        gc.collect()\n        __import__('ctypes').CDLL('libc.so.6').malloc_trim(0)\n        print(f'[mem] RSS={__import__(\\\"psutil\\\").Process().memory_info().rss/1e9:.2f}GB', flush=True)\n        \n    if torch.cuda.is_available():\";\
assert old in c, 'philosopher.py gc+malloc_trim patch target not found';\
c = c.replace(old, new);\
open('/ps/philosopher.py', 'w').write(c);\
print('philosopher.py gc+malloc_trim patch OK')\
"

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