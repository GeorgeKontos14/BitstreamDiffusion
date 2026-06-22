#!/bin/bash

pip install stable-audio-tools==0.0.17 --no-deps
pip install stable-codec==0.1.2 --no-deps

sed -i 's/^import k_diffusion as K$/K = None  # lazy: not needed for autoencoder encode\/decode/' \
  ~/.conda/envs/bitstream/lib/python3.10/site-packages/stable_audio_tools/inference/sampling.py

sed -i '1s/^from pytorch_lightning.loggers import WandbLogger, CometLogger$/pass  # lazy: not needed for encode\/decode/' \
  ~/.conda/envs/bitstream/lib/python3.10/site-packages/stable_audio_tools/training/utils.py

pip install alias-free-torch==0.0.6 --no-deps
pip install einops einops-exts --no-deps
pip install packaging scipy soundfile
pip install torchsde torchdiffeq trampoline --no-deps

pip install omegaconf einx

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'SparkAudio/Spark-TTS-0.5B',
    allow_patterns='BiCodec/*',
    local_dir='Spark-TTS/pretrained_models/SparkTTS-0.5B',
)
print('done')
"

pip install jiwer
pip install git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git --no-deps
pip install pysptk pyworld fastdtw