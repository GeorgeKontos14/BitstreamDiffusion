from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Tuple

import csv
import math

import numpy as np
import torch
from torch.amp import autocast
import torchaudio
from datasets import load_dataset, Audio
from tqdm import tqdm

# Spark-TTS is a git submodule; resolve relative to this file so the path is
# portable across machines and does not need to be hardcoded.
import sys
_SPARK_TTS_ROOT = Path(__file__).resolve().parents[2] / 'Spark-TTS'
if str(_SPARK_TTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARK_TTS_ROOT))

# ── vocab layout ─────────────────────────────────────────────────────────────

TEXT_VOCAB    = 200_019
SPEAKER_VOCAB = 4_096

TEXT_OFFSET    = 0
SPEAKER_OFFSET = TEXT_VOCAB          # 200_019
SPEECH_OFFSET  = TEXT_VOCAB + SPEAKER_VOCAB  # 204_115

# SPEECH_VOCAB, PAD_TOKEN, and TOTAL_VOCAB are bottleneck-dependent and
# computed at runtime in load_tokenizers / build_packed_cache.

MODEL_SR = 16_000  # both BiCodec and StableCodec operate at 16 kHz


# ── dataset / dataloader ──────────────────────────────────────────────────────

class CodecDataset(torch.utils.data.Dataset):
    """Wraps an HF dataset; decoding and resampling happen in DataLoader workers."""

    def __init__(self, hf_dataset, text_field: str, target_sr: int = MODEL_SR):
        self.ds = hf_dataset
        self.text_field = text_field
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.ds[idx]
        wav, sr = torchaudio.load(io.BytesIO(row['audio']['bytes']))
        wav = wav.mean(dim=0)  # mono, shape (T,)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav, row[self.text_field]


def _make_collate_fn(ds_ratio: int):
    """Returns a collate_fn that pads to the batch max aligned to ds_ratio."""
    def collate_fn(
        batch: list[Tuple[torch.Tensor, str]],
    ) -> Tuple[torch.Tensor, list[str], list[int]]:
        wavs, texts = zip(*batch)
        true_lengths = [w.shape[-1] for w in wavs]
        max_len = math.ceil(max(true_lengths) / ds_ratio) * ds_ratio
        padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) for w in wavs
        ]).unsqueeze(1)  # (B, 1, T_max)
        return padded, list(texts), true_lengths
    return collate_fn


# ── tokenization helpers ─────────────────────────────────────────────────────

def _tokenize_text_batch(
    text_tokenizer,
    texts: list[str],
    max_len: int,
    pad_token: int,
) -> Tuple[torch.Tensor, int]:
    """Returns ((B, max_len) int64 CPU PAD-filled, n_truncated)."""
    token_lists = text_tokenizer.encode_ordinary_batch(texts)
    batch = torch.full((len(texts), max_len), pad_token, dtype=torch.long)
    n_truncated = 0
    for i, toks in enumerate(token_lists):
        if len(toks) > max_len:
            n_truncated += 1
        toks = toks[:max_len]
        batch[i, :len(toks)] = torch.tensor(toks, dtype=torch.long) + TEXT_OFFSET
    return batch, n_truncated


@torch.no_grad()
def _tokenize_speaker_batch(
    bicodec,
    wav_batch: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    wav_batch: (B, 1, T) on device.
    Returns (B, seq_len) int64 on device. Always exactly seq_len tokens.
    """
    mel = bicodec.mel_transformer(wav_batch).squeeze(1)        # (B, n_mels, T_mel)
    with autocast('cuda', dtype=torch.float16):
        global_tokens = bicodec.speaker_encoder.tokenize(
            mel.transpose(1, 2)                                     # (B, T_mel, n_mels)
        )                                                           # (B, 1, seq_len)
    tokens = global_tokens.squeeze(1)                           # (B, seq_len)
    assert tokens.shape[1] == seq_len, (
        f'BiCodec speaker encoder produced {tokens.shape[1]} tokens, expected {seq_len}'
    )
    return tokens + SPEAKER_OFFSET


@torch.no_grad()
def _tokenize_speech_batch(
    stable_codec,
    wav_batch: torch.Tensor,
    true_wav_lengths: list[int],
    max_len: int,
    pad_token: int,
) -> Tuple[torch.Tensor, int]:
    """
    wav_batch: (B, 1, T) on device, already at MODEL_SR, zero-padded to batch max.
    true_wav_lengths: true waveform length in samples for each item, before padding.
    Returns ((B, max_len) int64 on device PAD-filled, n_truncated).

    pad_token is placed after each sample's true encoded length, not after the
    batch-padded length, so silence introduced by batch collation is never stored
    as real speech tokens.
    """
    normed = torch.stack(
        [stable_codec.volume_norm(wav_batch[i]) for i in range(wav_batch.shape[0])],
        dim=0,
    )
    ds = stable_codec.model.downsampling_ratio
    T = normed.shape[-1]
    pad = (ds - T % ds) % ds
    if pad > 0:
        normed = torch.nn.functional.pad(normed, (0, pad))
    with autocast('cuda', dtype=torch.float16):
        _, tokens = stable_codec.encode(normed, posthoc_bottleneck=True)
    tokens = tokens[0]                                          # (B, S, 1) or (B, S)
    if tokens.dim() == 3:
        tokens = tokens.squeeze(-1)                             # (B, S)

    result = torch.full(
        (tokens.shape[0], max_len), pad_token, dtype=torch.long, device=tokens.device,
    )
    n_truncated = 0
    for i, true_len in enumerate(true_wav_lengths):
        true_token_len = math.ceil(true_len / ds)
        if true_token_len > max_len:
            n_truncated += 1
        true_token_len = min(true_token_len, max_len)
        result[i, :true_token_len] = tokens[i, :true_token_len] + SPEECH_OFFSET
    return result, n_truncated


# ── duration filter ──────────────────────────────────────────────────────────

def load_valid_ids(duration_file: str, max_duration: float, max_samples: int = -1) -> set[str]:
    """Return the set of sample IDs whose duration <= max_duration seconds.

    Expected CSV format (with or without a header row):
        id, duration (seconds)
    """
    valid: set[str] = set()
    with open(duration_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            sample_id, duration_str = row[0].strip(), row[1].strip()
            try:
                if float(duration_str) <= max_duration:
                    valid.add(sample_id)
                    if max_samples > 0 and len(valid) >= max_samples:
                        break
            except ValueError:
                pass  # skip header or malformed rows
    return valid


# ── model loader ─────────────────────────────────────────────────────────────

def load_tokenizers(
    text_tokenizer_name: str,
    speaker_model_dir: str,
    speech_model: str,
    device: torch.device,
    speech_bottleneck: str | None = None,
    speech_bottleneck_dims: list[int] | None = None,
):
    """Load all three tokenizers and return (text_tok, bicodec, stable_codec, speech_vocab).

    Exactly one of speech_bottleneck (string preset) or speech_bottleneck_dims
    (list of per-codebook sizes, e.g. [8,8,8,8,8,8]) must be provided.
    speech_vocab is derived automatically: product of dims for the list form,
    or must be supplied explicitly for a string preset via speech_bottleneck_vocab.
    """
    if (speech_bottleneck is None) == (speech_bottleneck_dims is None):
        raise ValueError('Provide exactly one of speech_bottleneck or speech_bottleneck_dims')

    import tiktoken
    from sparktts.models.bicodec import BiCodec
    from stable_codec import StableCodec

    print(f'[libritts-cache] loading text tokenizer: {text_tokenizer_name}')
    text_tok = tiktoken.get_encoding(text_tokenizer_name)

    print(f'[libritts-cache] loading BiCodec from: {speaker_model_dir}')
    bicodec = BiCodec.load_from_checkpoint(model_dir=speaker_model_dir).to(device).eval()

    stable_codec = StableCodec(pretrained_model=speech_model, device=device)
    if speech_bottleneck_dims is not None:
        bottleneck_arg = [([speech_bottleneck_dims, 1.0])]
        speech_vocab = math.prod(speech_bottleneck_dims)
        print(f'[libritts-cache] loading StableCodec: {speech_model}  '
              f'bottleneck: dims={speech_bottleneck_dims}  speech_vocab={speech_vocab}')
    else:
        bottleneck_arg = speech_bottleneck
        speech_vocab = 46_656  # default for '1x46656_400bps'
        print(f'[libritts-cache] loading StableCodec: {speech_model}  bottleneck: {speech_bottleneck}')
    stable_codec.set_posthoc_bottleneck(bottleneck_arg)

    return text_tok, bicodec, stable_codec, speech_vocab


# ── cache builder ─────────────────────────────────────────────────────────────

def build_packed_cache(
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
    stable_codec,
    speech_seq_len: int,
    speech_vocab: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    valid_ids: set[str] | None,
    max_duration: float | None,
    text_field: str,
    force: bool,
    checkpoint_every: int = 50
) -> None:
    pad_token_text   = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text+1
    total_vocab = pad_token_speech + 1

    if cache_path.exists() and meta_path.exists() and not force:
        print(f'[libritts-cache] exists -> {cache_path}  (skip; pass --force to rebuild)')
        return

    seq_len = text_seq_len + speaker_seq_len + speech_seq_len

    if total_vocab > 2**32 - 1:
        raise RuntimeError(f'total_vocab={total_vocab} too large for uint32 memmap')

    # ── load HF dataset ──────────────────────────────────────────────────
    print(f'[libritts-cache] loading {hf_path!r} config={hf_config!r} split={hf_split!r}')
    dataset = load_dataset(
        hf_path, hf_config, split=hf_split,
    ).cast_column('audio', Audio(decode=False))

    if valid_ids is not None:
        before = len(dataset)
        dataset = dataset.filter(lambda x: x['id'] in valid_ids)
        print(f'[libritts-cache] duration filter (<= {max_duration}s): '
              f'{before:,} -> {len(dataset):,} samples kept')

    n_samples = len(dataset)
    print(f'[libritts-cache] {n_samples:,} samples  seq_len={seq_len} -> {cache_path.name}')

    # ── allocate memmap (atomic: write to .tmp, then rename) ─────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp')
    progress_path = cache_path.with_suffix('.progress')
    start_row = 0

    # if tmp_path.exists():
    #     print(f'[libritts-cache] removing stale tmp: {tmp_path}')
    #     tmp_path.unlink()

    # arr = np.memmap(tmp_path, dtype=np.uint32, mode='w+', shape=(n_samples, seq_len))

    if tmp_path.exists() and progress_path.exists() and not force:
        saved = json.loads(progress_path.read_text())
        if saved.get('n_samples') == n_samples:
            start_row = saved['row']
            print(f'[libritts-cache] resuming from row {start_row:,}/{n_samples:,}')
            arr = np.memmap(tmp_path, dtype=np.uint32, mode='r+',
                            shape=(n_samples, seq_len))
        else:
            print(f'[libritts-cache] stale checkpoint (n_samples mismatch), starting fresh')
            tmp_path.unlink()
            arr = np.memmap(tmp_path, dtype=np.uint32, mode='w+',
                            shape=(n_samples, seq_len))
    else:
        tmp_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        arr = np.memmap(tmp_path, dtype=np.uint32, mode='w+',
                        shape=(n_samples, seq_len))

    remaining = dataset.select(range(start_row, n_samples)) if start_row > 0 else dataset

    ds_ratio = stable_codec.model.downsampling_ratio
    loader = torch.utils.data.DataLoader(
        CodecDataset(remaining, text_field),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_make_collate_fn(ds_ratio),
        pin_memory=(device.type == 'cuda'),
    )

    n_text_truncated = 0
    n_speech_truncated = 0
    row = start_row

    try:
        for batch_idx, (wav_batch, texts, true_wav_lengths) in enumerate(tqdm(
            loader,
            desc=f'Encoding {split_name}',
            initial=start_row // batch_size,
            total = math.ceil(n_samples / batch_size),
            dynamic_ncols=True,
        )):
            wav_batch = wav_batch.to(device, non_blocking=True)

            text_tokens,    n_txt_trunc  = _tokenize_text_batch(text_tokenizer, texts, text_seq_len, pad_token_text)
            speaker_tokens               = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len)
            speech_tokens,  n_sp_trunc   = _tokenize_speech_batch(stable_codec, wav_batch, true_wav_lengths, speech_seq_len, pad_token_speech)

            n_text_truncated   += n_txt_trunc
            n_speech_truncated += n_sp_trunc

            combined = torch.cat(
                [text_tokens.to(device), speaker_tokens, speech_tokens], dim=1,
            ).cpu().numpy().astype(np.uint32)  # (B, seq_len)

            arr[row:row + combined.shape[0]] = combined
            row += combined.shape[0]

            if (batch_idx+1)%checkpoint_every == 0:
                arr.flush()
                progress_path.write_text(json.dumps({
                    'row': row,
                    'n_samples': n_samples,
                    'n_text_truncated': n_text_truncated,
                    'n_speech_truncated': n_speech_truncated
                }))

        arr.flush()
        del arr
        os.replace(tmp_path, cache_path)

    except Exception:
        try:
            arr.flush()
            progress_path.write_text(json.dumps({
                'row': row,
                'n_samples': n_samples,
                'n_text_truncated': n_text_truncated,
                'n_speech_truncated': n_speech_truncated
            }))            
            del arr
        except Exception:
            pass
        raise

    meta = {
        'cache_format': 'packed_multimodal_blocks',
        'dtype': 'uint32',
        'hf_path': hf_path,
        'hf_config': hf_config,
        'hf_split': hf_split,
        'split_name': split_name,
        'n_sequences': n_samples,
        'seq_len_tokens': seq_len,
        'total_vocab': total_vocab,
        'pad_token_text': pad_token_text,
        'pad_token_spech': pad_token_speech,
        'text_seq_len': text_seq_len,
        'text_offset': TEXT_OFFSET,
        'text_vocab': TEXT_VOCAB,
        'n_text_truncated': n_text_truncated,
        'speaker_seq_len': speaker_seq_len,
        'speaker_offset': SPEAKER_OFFSET,
        'speaker_vocab': SPEAKER_VOCAB,
        'speech_seq_len': speech_seq_len,
        'speech_offset': SPEECH_OFFSET,
        'speech_vocab': speech_vocab,
        'n_speech_truncated': n_speech_truncated,
        'max_duration': max_duration,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'[libritts-cache] wrote {n_samples:,} × {seq_len} -> {cache_path}')
    print(f'[libritts-cache] text_truncated={n_text_truncated:,}  '
          f'speech_truncated={n_speech_truncated:,}')
    print(f'[libritts-cache] meta -> {meta_path}')


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build fixed-length multimodal token caches for LibriTTS.',
    )
    # ── LibriTTS (train/val) ─────────────────────────────────────────────────
    ap.add_argument('--hf_path',      type=str, default='mythicinfinity/libritts')
    ap.add_argument('--hf_config',    type=str, default='all')
    ap.add_argument('--train_split',  type=str,
                    default='train.clean.100+train.clean.360+train.other.500')
    ap.add_argument('--train_duration_file', type=str, default='durations/libri_train.csv',
                    help='CSV file (id, duration) for the training split.')
    ap.add_argument('--val_split', type=str, default='dev.clean')
    ap.add_argument('--val_duration_file', type=str, default='durations/libri_val.csv')
    ap.add_argument('--val_size', type=int, default=512)
    # ── LibriSpeech (test) ───────────────────────────────────────────────
    ap.add_argument('--test_hf_path',   type=str,
                    default='mythicinfinity/librispeech-pc-44khz-opus')
    ap.add_argument('--test_clean_config', type=str, default='clean')
    ap.add_argument('--test_clean_split',  type=str, default='test')
    ap.add_argument('--test_other_config', type=str, default='other')
    ap.add_argument('--test_other_split',  type=str, default='test')
    ap.add_argument('--test_clean_duration_file', type=str, default='durations/libri_test_clean.csv',
                    help='CSV file (id, duration) for test.clean.')
    ap.add_argument('--test_other_duration_file', type=str, default='durations/libri_test_other.csv',
                    help='CSV file (id, duration) for test.other.')

    ap.add_argument('--out_dir',      type=str, default='datasets/libri')
    ap.add_argument('--max_duration', type=float, default=32.0,
                    help='Drop samples longer than this many seconds (default: 32.0).')

    ap.add_argument('--text_tokenizer',    type=str, default='o200k_base')
    ap.add_argument('--text_seq_len',      type=int, default=168)
    ap.add_argument('--speaker_model_dir', type=str,
                    default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speaker_seq_len',   type=int, default=32)
    ap.add_argument('--speech_model',      type=str,
                    default='stabilityai/stable-codec-speech-16k')
    bn_group = ap.add_mutually_exclusive_group(required=False)
    bn_group.add_argument('--speech_bottleneck', type=str, default=None,
                          help='String preset passed to StableCodec.set_posthoc_bottleneck '
                               '(e.g. "1x46656_400bps", speech_vocab=46656).')
    bn_group.add_argument('--speech_bottleneck_dims', type=int, nargs='+', metavar='N',
                          default=None,
                          help='Per-codebook sizes for the list-form bottleneck '
                               '(e.g. 8 8 8 8 8 8 → speech_vocab=2^18=262144).')
    ap.add_argument('--speech_seq_len',    type=int, default=800)

    ap.add_argument('--batch_size',  type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4,
                    help='DataLoader worker processes for parallel audio decoding.')
    ap.add_argument('--device',      type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--force',      action='store_true')
    ap.add_argument('--checkpoint_every', type=int, default=50)
    args = ap.parse_args()

    # Apply default bottleneck if neither flag was given.
    if args.speech_bottleneck is None and args.speech_bottleneck_dims is None:
        args.speech_bottleneck = '1x46656_400bps'

    out_dir = Path(args.out_dir)
    device  = torch.device(args.device)

    text_tokenizer, bicodec, stable_codec, speech_vocab = load_tokenizers(
        text_tokenizer_name=args.text_tokenizer,
        speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        device=device,
        speech_bottleneck=args.speech_bottleneck,
        speech_bottleneck_dims=args.speech_bottleneck_dims,
    )

    def _load_ids(duration_file: str | None, max_samples: int = -1) -> set[str] | None:
        if duration_file is None:
            return None
        ids = load_valid_ids(duration_file, args.max_duration, max_samples)
        print(f'[libritts-cache] {len(ids):,} IDs pass duration filter '
              f'(<= {args.max_duration}s) from {duration_file}')
        return ids

    train_valid_ids      = _load_ids(args.train_duration_file)
    val_valid_ids        = _load_ids(args.val_duration_file, args.val_size)
    test_clean_valid_ids = _load_ids(args.test_clean_duration_file)
    test_other_valid_ids = _load_ids(args.test_other_duration_file)

    # Parameters shared across all four splits.
    shared = dict(
        text_tokenizer=text_tokenizer,
        text_seq_len=args.text_seq_len,
        bicodec=bicodec,
        speaker_seq_len=args.speaker_seq_len,
        stable_codec=stable_codec,
        speech_seq_len=args.speech_seq_len,
        speech_vocab=speech_vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_duration=args.max_duration,
        force=args.force,
        checkpoint_every=args.checkpoint_every
    )

    build_packed_cache(
        hf_path=args.hf_path,
        hf_config=args.hf_config,
        hf_split=args.train_split,
        split_name='train',
        cache_path=out_dir / 'cache_libri_train.uint32',
        meta_path=out_dir / 'cache_libri_train.meta.json',
        valid_ids=train_valid_ids,
        text_field='text_normalized',
        **shared,
    )

    build_packed_cache(
        hf_path=args.hf_path,
        hf_config=args.hf_config,
        hf_split=args.val_split,
        split_name='val',
        cache_path=out_dir / 'cache_libri_val.uint32',
        meta_path=out_dir / 'cache_libri_val.meta.json',
        valid_ids=val_valid_ids,
        text_field='text_normalized',
        **shared,
    )


    build_packed_cache(
        hf_path=args.test_hf_path,
        hf_config=args.test_clean_config,
        hf_split=args.test_clean_split,
        split_name='test_clean',
        cache_path=out_dir / 'cache_libri_test_clean.uint32',
        meta_path=out_dir / 'cache_libri_test_clean.meta.json',
        valid_ids=test_clean_valid_ids,
        text_field='text',
        **shared,
    )

    build_packed_cache(
        hf_path=args.test_hf_path,
        hf_config=args.test_other_config,
        hf_split=args.test_other_split,
        split_name='test_other',
        cache_path=out_dir / 'cache_libri_test_other.uint32',
        meta_path=out_dir / 'cache_libri_test_other.meta.json',
        valid_ids=test_other_valid_ids,
        text_field='text',
        **shared,
    )


if __name__ == '__main__':
    main()
