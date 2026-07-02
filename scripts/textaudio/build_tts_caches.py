from __future__ import annotations

import argparse
import io
import json
import os

from pathlib import Path

import csv

import numpy as np
import torch
import torchaudio
from datasets import load_dataset, Audio
from tqdm import tqdm

import sys
_SPARK_TTS_ROOT = Path(__file__).resolve().parents[2] / 'Spark-TTS'
if str(_SPARK_TTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARK_TTS_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_textaudio_caches import (
    TEXT_VOCAB,
    SPEAKER_VOCAB,
    TEXT_OFFSET,
    SPEAKER_OFFSET,
    SPEECH_OFFSET,
    MODEL_SR,
    load_valid_ids,
    _tokenize_text_batch,
    _tokenize_speaker_batch,
)

# -----------------------------------------------------------------------------
# Vocabulary layout
# -----------------------------------------------------------------------------
SPEECH_VOCAB = 46_656

# -----------------------------------------------------------------------------
# Loading methods
# -----------------------------------------------------------------------------

def load_heldout_map(heldout_file: str) -> dict[str, str]:
    """Returns speaker_id -> held-out sample id.

    Expected CSV format (no header): speaker_id, sample_id
    The held-out sample's audio is the only source of speaker tokens for that
    speaker; the sample itself is excluded from the cache to avoid leakage.
    """
    mapping: dict[str, str] = {}
    with open(heldout_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            speaker_id, sample_id = row[0].strip(), row[1].strip()
            mapping[speaker_id] = sample_id
    return mapping


def load_text_and_speaker_tokenizers(
    text_tokenizer_name: str,
    speaker_model_dir: str,
    device: torch.device,
):
    """Loads only the text tokenizer and BiCodec -- no StableCodec, since this
    cache never stores speech tokens."""
    import tiktoken
    from sparktts.models.bicodec import BiCodec

    print(f'[tts-cache] loading text tokenizer: {text_tokenizer_name}')
    text_tok = tiktoken.get_encoding(text_tokenizer_name)

    print(f'[tts-cache] loading BiCodec from: {speaker_model_dir}')
    bicodec = BiCodec.load_from_checkpoint(model_dir=speaker_model_dir).to(device).eval()

    return text_tok, bicodec


def _load_and_pad_wavs(rows, target_sr: int) -> torch.Tensor:
    """Decodes+resamples a handful of HF dataset rows and zero-pads to a common
    length. Only used for the held-out (one-per-speaker) audios, so a plain
    synchronous loop is plenty -- no DataLoader/worker pool needed."""
    wavs = []
    for row in rows:
        wav, sr = torchaudio.load(io.BytesIO(row['audio']['bytes']))
        wav = wav.mean(dim=0)  # mono, shape (T,)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wavs.append(wav)
    max_len = max(w.shape[-1] for w in wavs)
    return torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) for w in wavs
    ]).unsqueeze(1)  # (B, 1, T_max)


# -----------------------------------------------------------------------------
# Cache builder for Text-to-speech evaluation
# Text tokenization is performed per-sample
# Speaker tokens are produced from the speaker's held out sample
# -----------------------------------------------------------------------------

def build_tts_cache(
    *,
    hf_path: str,
    hf_config: str,
    hf_split: str,
    split_name: str,
    cache_path: Path,
    meta_path: Path,
    text_tokenizer,
    text_seq_len: int,
    bicodec,
    speaker_seq_len: int,
    speech_vocab: int,
    device: torch.device,
    valid_ids: set[str] | None,
    max_duration: float | None,
    text_field: str,
    heldout_map: dict[str, str],
    force: bool,
    text_batch_size: int = 1024,
) -> None:
    pad_token_text   = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1
    seq_len = text_seq_len + speaker_seq_len

    if cache_path.exists() and meta_path.exists() and not force:
        print(f'[tts-cache] exists -> {cache_path}  (skip; pass --force to rebuild)')
        return

    print(f'[tts-cache] loading {hf_path!r} config={hf_config!r} split={hf_split!r}')
    dataset = load_dataset(
        hf_path, hf_config, split=hf_split,
    ).cast_column('audio', Audio(decode=False))

    if valid_ids is not None:
        before = len(dataset)
        dataset = dataset.filter(lambda x: x['id'] in valid_ids)
        print(f'[tts-cache] duration filter (<= {max_duration}s): '
              f'{before:,} -> {len(dataset):,} samples kept')

    heldout_sample_ids = set(heldout_map.values())

    heldout_rows = dataset.filter(lambda x: x['id'] in heldout_sample_ids)
    rest = dataset.filter(lambda x: x['id'] not in heldout_sample_ids)

    heldout_speaker_ids = [str(s) for s in heldout_rows['speaker_id']]
    missing_speakers = [spk for spk in heldout_map if spk not in heldout_speaker_ids]
    if missing_speakers:
        print(f'[tts-cache] WARNING: {len(missing_speakers)} held-out samples missing '
              f'(filtered out or absent) -> those speakers will be dropped entirely: '
              f'{missing_speakers}')

    print(f'[tts-cache] encoding speaker tokens for {len(heldout_rows):,} held-out speakers')
    wav_batch = _load_and_pad_wavs(heldout_rows, MODEL_SR).to(device, non_blocking=True)
    speaker_tok_batch = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len).cpu()

    speaker_tokens: dict[str, torch.Tensor] = dict(zip(heldout_speaker_ids, speaker_tok_batch))

    rest_texts = rest[text_field]
    rest_speaker_ids = [str(s) for s in rest['speaker_id']]

    kept_indices = [i for i, spk in enumerate(rest_speaker_ids) if spk in speaker_tokens]
    n_dropped = len(rest_speaker_ids) - len(kept_indices)
    if n_dropped:
        print(f'[tts-cache] dropping {n_dropped:,} rows whose speaker has no usable '
              f'held-out sample')

    kept_texts = [rest_texts[i] for i in kept_indices]
    kept_speaker_ids = [rest_speaker_ids[i] for i in kept_indices]
    n_samples = len(kept_indices)
    print(f'[tts-cache] {n_samples:,} samples  seq_len={seq_len} -> {cache_path.name}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp')
    arr = np.memmap(tmp_path, dtype=np.uint32, mode='w+', shape=(n_samples, seq_len))

    n_text_truncated = 0
    for start in tqdm(
        range(0, n_samples, text_batch_size),
        desc=f'Encoding {split_name}',
        total=-(-n_samples // text_batch_size),
        dynamic_ncols=True,
    ):
        chunk_texts    = kept_texts[start:start + text_batch_size]
        chunk_speakers = kept_speaker_ids[start:start + text_batch_size]

        text_tokens, n_trunc = _tokenize_text_batch(text_tokenizer, chunk_texts, text_seq_len, pad_token_text)
        n_text_truncated += n_trunc

        speaker_block = torch.stack([speaker_tokens[s] for s in chunk_speakers], dim=0)
        combined = torch.cat([text_tokens, speaker_block], dim=1).numpy().astype(np.uint32)
        arr[start:start + combined.shape[0]] = combined

    arr.flush()
    del arr
    os.replace(tmp_path, cache_path)

    meta = {
        'cache_format': 'packed_tts_blocks',
        'dtype': 'uint32',
        'hf_path': hf_path,
        'hf_config': hf_config,
        'hf_split': hf_split,
        'split_name': split_name,
        'n_sequences': n_samples,
        'seq_len_tokens': seq_len,
        'total_vocab': total_vocab,
        'pad_token_text': pad_token_text,
        'pad_token_speech': pad_token_speech,
        'text_seq_len': text_seq_len,
        'text_offset': TEXT_OFFSET,
        'text_vocab': TEXT_VOCAB,
        'n_text_truncated': n_text_truncated,
        'speaker_seq_len': speaker_seq_len,
        'speaker_offset': SPEAKER_OFFSET,
        'speaker_vocab': SPEAKER_VOCAB,
        'speech_offset': SPEECH_OFFSET,
        'speech_vocab': speech_vocab,
        'max_duration': max_duration,
        'n_speakers': len(speaker_tokens),
        'n_speakers_dropped': len(missing_speakers),
        'n_rows_dropped_missing_speaker': n_dropped,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'[tts-cache] wrote {n_samples:,} × {seq_len} -> {cache_path}')
    print(f'[tts-cache] text_truncated={n_text_truncated:,}  speakers={len(speaker_tokens):,}')
    print(f'[tts-cache] meta -> {meta_path}')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build text+speaker (no speech-token) TTS caches from '
                     'LibriTTS dev.clean and LibriSpeech-PC test.clean, with '
                     'per-speaker held-out samples for leakage-free speaker tokens.',
    )
    ap.add_argument('--libritts_hf_path',   type=str, default='mythicinfinity/libritts')
    ap.add_argument('--libritts_hf_config', type=str, default='clean')
    ap.add_argument('--libritts_hf_split',  type=str, default='dev.clean')
    ap.add_argument('--libritts_duration_file', type=str, default='assets/durations/val.csv')
    ap.add_argument('--libritts_heldout_file',  type=str, default='assets/heldout/val.csv')

    ap.add_argument('--librispeech_hf_path',   type=str, default='mythicinfinity/librispeech-pc-44khz-opus')
    ap.add_argument('--librispeech_hf_config', type=str, default='clean')
    ap.add_argument('--librispeech_hf_split',  type=str, default='test')
    ap.add_argument('--librispeech_duration_file', type=str, default='assets/durations/test_clean.csv')
    ap.add_argument('--librispeech_heldout_file',  type=str, default='assets/heldout/test_clean.csv')

    ap.add_argument('--out_dir',      type=str, default='datasets/tts')
    ap.add_argument('--max_duration', type=float, default=32.0,
                    help='Drop samples longer than this many seconds (default: 32.0).')

    ap.add_argument('--text_tokenizer',    type=str, default='o200k_base')
    ap.add_argument('--text_seq_len',      type=int, default=168)
    ap.add_argument('--speaker_model_dir', type=str,
                    default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speaker_seq_len',   type=int, default=32)

    ap.add_argument('--text_batch_size', type=int, default=1024,
                    help='Chunk size for batched text tokenization.')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    device  = torch.device(args.device)

    text_tokenizer, bicodec = load_text_and_speaker_tokenizers(
        text_tokenizer_name=args.text_tokenizer,
        speaker_model_dir=args.speaker_model_dir,
        device=device,
    )

    libritts_valid_ids    = load_valid_ids(args.libritts_duration_file, args.max_duration)
    librispeech_valid_ids = load_valid_ids(args.librispeech_duration_file, args.max_duration)
    print(f'[tts-cache] {len(libritts_valid_ids):,} LibriTTS IDs pass duration filter '
          f'(<= {args.max_duration}s)')
    print(f'[tts-cache] {len(librispeech_valid_ids):,} LibriSpeech IDs pass duration filter '
          f'(<= {args.max_duration}s)')

    libritts_heldout    = load_heldout_map(args.libritts_heldout_file)
    librispeech_heldout = load_heldout_map(args.librispeech_heldout_file)

    shared = dict(
        text_tokenizer=text_tokenizer,
        text_seq_len=args.text_seq_len,
        bicodec=bicodec,
        speaker_seq_len=args.speaker_seq_len,
        speech_vocab=SPEECH_VOCAB,
        device=device,
        max_duration=args.max_duration,
        force=args.force,
        text_batch_size=args.text_batch_size,
    )

    build_tts_cache(
        hf_path=args.libritts_hf_path,
        hf_config=args.libritts_hf_config,
        hf_split=args.libritts_hf_split,
        split_name='libritts_dev_clean',
        cache_path=out_dir / 'cache_dev_clean_tts.uint32',
        meta_path=out_dir / 'cache_dev_clean_tts.meta.json',
        valid_ids=libritts_valid_ids,
        text_field='text_normalized',
        heldout_map=libritts_heldout,
        **shared,
    )

    build_tts_cache(
        hf_path=args.librispeech_hf_path,
        hf_config=args.librispeech_hf_config,
        hf_split=args.librispeech_hf_split,
        split_name='librispeech_test_clean',
        cache_path=out_dir / 'cache_test_clean_tts.uint32',
        meta_path=out_dir / 'cache_test_clean_tts.meta.json',
        valid_ids=librispeech_valid_ids,
        text_field='text',
        heldout_map=librispeech_heldout,
        **shared,
    )


if __name__ == '__main__':
    main()
