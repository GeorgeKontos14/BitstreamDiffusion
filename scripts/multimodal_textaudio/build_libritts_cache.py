from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Audio

from cache_utils import build_packed_cache, load_valid_ids

HF_PATH = 'mythicinfinity/libritts'
HF_CONFIG = 'all'
TRAIN_SPLITS = ['train.clean.100', 'train.clean.360', 'train.other.500']
TEXT_FIELD = 'text_normalized'

OUT_DIR = Path('datasets/libri')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default=str(OUT_DIR))
    ap.add_argument('--duration_file', type=str, default='assets/durations/train.csv')
    ap.add_argument('--max_duration', type=float, default=32.0)

    ap.add_argument('--text_tokenizer', type=str, default='o200k_base')
    ap.add_argument('--text_seq_len', type=int, default=100)
    ap.add_argument('--speaker_model_dir', type=str,
                     default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speaker_seq_len', type=int, default=32)
    ap.add_argument('--speech_model', type=str, default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--speech_seq_len', type=int, default=500)

    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=32)
    ap.add_argument('--gpus', type=int, nargs='*', default=None,
                     help='GPU ids to use; omit for all visible GPUs.')
    ap.add_argument('--checkpoint_every', type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        import torch
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]

    valid_ids = load_valid_ids(args.duration_file, args.max_duration)
    print(f'[libri-cache] {len(valid_ids):,} ids pass duration filter (<= {args.max_duration}s)')

    print(f'[libri-cache] loading {HF_PATH!r} config={HF_CONFIG!r} splits={TRAIN_SPLITS}')
    ds = load_dataset(HF_PATH, HF_CONFIG)
    dataset = concatenate_datasets([ds[s] for s in TRAIN_SPLITS]).cast_column('audio', Audio(decode=False))
    dataset = dataset.filter(lambda x: x['id'] in valid_ids)
    print(f'[libri-cache] {len(dataset):,} samples after filtering, workers: {gpu_ids}')

    build_packed_cache(
        dataset=dataset,
        cache_path=out_dir / 'cache_libri_train.uint32',
        meta_path=out_dir / 'cache_libri_train.meta.json',
        hf_path=HF_PATH, hf_config=HF_CONFIG, hf_split='+'.join(TRAIN_SPLITS),
        split_name='libri_train',
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
        max_duration=args.max_duration,
        row_durations=None,
    )


if __name__ == '__main__':
    main()
