from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset, Audio

from cache_utils import build_packed_cache

HF_PATH = 'parler-tts/mls_eng'
HF_SPLIT = 'train'
TEXT_FIELD = 'transcript'

OUT_DIR = Path('datasets/mls')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default=str(OUT_DIR))

    ap.add_argument('--text_tokenizer', type=str, default='o200k_base')
    ap.add_argument('--text_seq_len', type=int, default=100)
    ap.add_argument('--speaker_model_dir', type=str,
                     default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speaker_seq_len', type=int, default=32)
    ap.add_argument('--speech_model', type=str, default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--speech_seq_len', type=int, default=500)

    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--gpus', type=int, nargs='*', default=None,
                     help='GPU ids to use; omit for all visible GPUs, or pass '
                          'no values at all to force a single CPU process.')
    ap.add_argument('--checkpoint_every', type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        import torch
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]

    print(f'[mls-cache] loading {HF_PATH!r} split={HF_SPLIT!r}')
    dataset = load_dataset(HF_PATH, split=HF_SPLIT).cast_column('audio', Audio(decode=False))
    print(f'[mls-cache] {len(dataset):,} samples, workers: {gpu_ids}')

    row_durations = np.asarray(dataset['audio_duration'], dtype=np.float64)

    build_packed_cache(
        dataset=dataset,
        cache_path=out_dir / 'cache_mls_train.uint32',
        meta_path=out_dir / 'cache_mls_train.meta.json',
        hf_path=HF_PATH, hf_config=None, hf_split=HF_SPLIT,
        split_name='mls_train',
        text_field=TEXT_FIELD,
        text_tokenizer_name=args.text_tokenizer,
        speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        text_seq_len=args.text_seq_len,
        speaker_seq_len=args.speaker_seq_len,
        speech_seq_len=args.speech_seq_len,
        gpu_ids=gpu_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        max_duration=None,
        row_durations=row_durations,
    )


if __name__ == '__main__':
    main()
