from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset, Audio

from cache_utils import (
    load_valid_ids, load_heldout_map,
    build_asr_cache, build_tts_cache, build_continuation_cache,
)

HF_PATH = 'mythicinfinity/libritts'
HF_CONFIG = 'all'
HF_SPLIT = 'dev.clean'
TEXT_FIELD = 'text_normalized'

OUT_DIR = Path('datasets/validation')
CONT_TRIM_SECONDS = 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--duration_file', type=str, default='assets/durations/val.csv')
    ap.add_argument('--heldout_file', type=str, default='assets/heldout/val.csv')
    ap.add_argument('--max_duration', type=float, default=32.0)

    ap.add_argument('--text_seq_len', type=int, default=100)
    ap.add_argument('--speaker_seq_len', type=int, default=32)
    ap.add_argument('--speech_seq_len', type=int, default=500)

    ap.add_argument('--text_tokenizer', type=str, default='o200k_base')
    ap.add_argument('--speaker_model_dir', type=str,
                     default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speech_model', type=str, default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_dir', type=str, default=str(OUT_DIR))
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    print(f'[val] loading {HF_PATH!r} config={HF_CONFIG!r} split={HF_SPLIT!r}')
    dataset = load_dataset(HF_PATH, HF_CONFIG, split=HF_SPLIT).cast_column('audio', Audio(decode=False))

    heldout_map = load_heldout_map(args.heldout_file)

    asr_valid_ids = load_valid_ids(args.duration_file, args.max_duration)
    build_asr_cache(
        dataset=dataset, out_dir=out_dir, stem='cache_val',
        hf_path=HF_PATH, hf_config=HF_CONFIG, hf_split=HF_SPLIT, split_name='val',
        text_field=TEXT_FIELD, valid_ids=asr_valid_ids,
        text_tokenizer_name=args.text_tokenizer, speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        text_seq_len=args.text_seq_len, speaker_seq_len=args.speaker_seq_len,
        speech_seq_len=args.speech_seq_len, device=device, batch_size=args.batch_size,
        size=args.size, max_duration=args.max_duration,
    )

    tts_valid_ids = load_valid_ids(args.duration_file, args.max_duration)
    build_tts_cache(
        dataset=dataset, out_dir=out_dir, stem='cache_val',
        hf_path=HF_PATH, hf_config=HF_CONFIG, hf_split=HF_SPLIT, split_name='val',
        text_field=TEXT_FIELD, valid_ids=tts_valid_ids, heldout_map=heldout_map,
        text_tokenizer_name=args.text_tokenizer, speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        text_seq_len=args.text_seq_len, speaker_seq_len=args.speaker_seq_len,
        speech_seq_len_for_suffix=args.speech_seq_len, device=device, batch_size=args.batch_size,
        size=args.size,
    )

    cont_valid_ids = load_valid_ids(args.duration_file, args.max_duration, min_duration=CONT_TRIM_SECONDS)
    build_continuation_cache(
        dataset=dataset, out_dir=out_dir, stem='cache_val',
        hf_path=HF_PATH, hf_config=HF_CONFIG, hf_split=HF_SPLIT, split_name='val',
        valid_ids=cont_valid_ids,
        speaker_model_dir=args.speaker_model_dir, speech_model=args.speech_model,
        speaker_seq_len=args.speaker_seq_len, device=device, batch_size=args.batch_size,
        size=args.size, trim_seconds=CONT_TRIM_SECONDS, max_duration=args.max_duration,
    )


if __name__ == '__main__':
    main()
