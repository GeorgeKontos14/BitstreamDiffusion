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
import torch.multiprocessing as mp
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

# -----------------------------------------------------------------------------
# Vocabulary layout
# -----------------------------------------------------------------------------

TEXT_VOCAB    = 200_019
SPEAKER_VOCAB = 4_096

TEXT_OFFSET    = 0
SPEAKER_OFFSET = TEXT_VOCAB          # 200_019
SPEECH_OFFSET  = TEXT_VOCAB + SPEAKER_VOCAB  # 204_115

# SPEECH_VOCAB, PAD_TOKEN, and TOTAL_VOCAB are bottleneck-dependent and
# computed at runtime in compute_speech_vocab / build_packed_cache.

MODEL_SR = 16_000  # both BiCodec and StableCodec operate at 16 kHz


# -----------------------------------------------------------------------------
# Dataloader construction
# -----------------------------------------------------------------------------

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


class _CollateFn:
    """Pads a batch to its max length aligned to ds_ratio.

    A top-level class rather than a closure: DataLoader workers spawned from
    inside an already-mp.spawn'd GPU worker process must pickle the collate_fn,
    and closures aren't picklable.
    """

    def __init__(self, ds_ratio: int):
        self.ds_ratio = ds_ratio

    def __call__(
        self,
        batch: list[Tuple[torch.Tensor, str]],
    ) -> Tuple[torch.Tensor, list[str], list[int]]:
        wavs, texts = zip(*batch)
        true_lengths = [w.shape[-1] for w in wavs]
        max_len = math.ceil(max(true_lengths) / self.ds_ratio) * self.ds_ratio
        padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) for w in wavs
        ]).unsqueeze(1)  # (B, 1, T_max)
        return padded, list(texts), true_lengths


# -----------------------------------------------------------------------------
# Tokenization logic
# Separate pad tokens are used for text and speech
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Duration filtering: only samples in [min_duration, max_duration]=[0,32] 
# are maintained
# -----------------------------------------------------------------------------

def load_valid_ids(
    duration_file: str,
    max_duration: float,
    max_samples: int = -1,
    min_duration: float = 0.0,
) -> set[str]:
    """Return the set of sample IDs whose duration is in [min_duration, max_duration] seconds.

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
                if min_duration <= float(duration_str) <= max_duration:
                    valid.add(sample_id)
                    if max_samples > 0 and len(valid) >= max_samples:
                        break
            except ValueError:
                pass  # skip header or malformed rows
    return valid

# -----------------------------------------------------------------------------
# Tokenizer loading
# -----------------------------------------------------------------------------

def compute_speech_vocab(
    speech_bottleneck: str | None,
    speech_bottleneck_dims: list[int] | None,
) -> int:
    """Derive speech_vocab without loading any heavy model (cheap, used for sizing)."""
    if speech_bottleneck_dims is not None:
        return math.prod(speech_bottleneck_dims)
    return 46_656  # default for '1x46656_400bps'


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

    print(f'[textaudio-cache] loading text tokenizer: {text_tokenizer_name}')
    text_tok = tiktoken.get_encoding(text_tokenizer_name)

    print(f'[textaudio-cache] loading BiCodec from: {speaker_model_dir}')
    bicodec = BiCodec.load_from_checkpoint(model_dir=speaker_model_dir).to(device).eval()

    stable_codec = StableCodec(pretrained_model=speech_model, device=device)
    speech_vocab = compute_speech_vocab(speech_bottleneck, speech_bottleneck_dims)
    if speech_bottleneck_dims is not None:
        bottleneck_arg = [([speech_bottleneck_dims, 1.0])]
        print(f'[textaudio-cache] loading StableCodec: {speech_model}  '
              f'bottleneck: dims={speech_bottleneck_dims}  speech_vocab={speech_vocab}')
    else:
        bottleneck_arg = speech_bottleneck
        print(f'[textaudio-cache] loading StableCodec: {speech_model}  bottleneck: {speech_bottleneck}')
    stable_codec.set_posthoc_bottleneck(bottleneck_arg)

    return text_tok, bicodec, stable_codec, speech_vocab


# -----------------------------------------------------------------------------
# Multi-node logic
# -----------------------------------------------------------------------------

def _merge_stats(stats_dir: Path, cum_text: int, cum_speech: int, trunc_path: Path) -> Tuple[int, int]:
    """Fold every rank's this-run stats file into the durable cumulative total."""
    if stats_dir.exists():
        for p in sorted(stats_dir.glob('rank*.json')):
            try:
                d = json.loads(p.read_text())
                cum_text   += d.get('n_text_truncated', 0)
                cum_speech += d.get('n_speech_truncated', 0)
            except (json.JSONDecodeError, OSError):
                pass
            p.unlink(missing_ok=True)
        try:
            stats_dir.rmdir()
        except OSError:
            pass  # not empty / already gone; harmless
    trunc_path.write_text(json.dumps({
        'n_text_truncated': cum_text,
        'n_speech_truncated': cum_speech,
    }))
    return cum_text, cum_speech


def _worker_main(
    local_rank: int,
    gpu_ids: list[int | None],
    index_chunks: list[np.ndarray],
    dataset,
    text_field: str,
    text_tokenizer_name: str,
    speaker_model_dir: str,
    speech_model: str,
    speech_bottleneck: str | None,
    speech_bottleneck_dims: list[int] | None,
    tmp_path: Path,
    done_path: Path,
    n_samples: int,
    seq_len: int,
    text_seq_len: int,
    speaker_seq_len: int,
    speech_seq_len: int,
    pad_token_text: int,
    pad_token_speech: int,
    batch_size: int,
    num_workers: int,
    checkpoint_every: int,
    stats_dir: Path,
    split_name: str,
) -> None:
    """Encodes this rank's assigned dataset rows and writes them straight into the
    shared memmap at their absolute row indices, so any number of ranks (this run
    or a future one) can pick up whichever rows are still unmarked in `done_path`.
    """
    indices = index_chunks[local_rank]
    if len(indices) == 0:
        return

    gpu_id = gpu_ids[local_rank]
    device = torch.device('cpu') if gpu_id is None else torch.device(f'cuda:{gpu_id}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    text_tok, bicodec, stable_codec, _ = load_tokenizers(
        text_tokenizer_name=text_tokenizer_name,
        speaker_model_dir=speaker_model_dir,
        speech_model=speech_model,
        device=device,
        speech_bottleneck=speech_bottleneck,
        speech_bottleneck_dims=speech_bottleneck_dims,
    )

    arr  = np.memmap(tmp_path,  dtype=np.uint32, mode='r+', shape=(n_samples, seq_len))
    done = np.memmap(done_path, dtype=np.uint8,  mode='r+', shape=(n_samples,))

    subset = dataset.select(indices.tolist())
    ds_ratio = stable_codec.model.downsampling_ratio
    loader = torch.utils.data.DataLoader(
        CodecDataset(subset, text_field),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_CollateFn(ds_ratio),
        pin_memory=(device.type == 'cuda'),
    )

    n_text_truncated = 0
    n_speech_truncated = 0
    cursor = 0
    stats_path = stats_dir / f'rank{local_rank}.json'

    def _checkpoint() -> None:
        arr.flush()
        done.flush()
        stats_path.write_text(json.dumps({
            'n_text_truncated': n_text_truncated,
            'n_speech_truncated': n_speech_truncated,
        }))

    try:
        for batch_idx, (wav_batch, texts, true_wav_lengths) in enumerate(tqdm(
            loader,
            desc=f'{split_name} rank{local_rank}',
            total=math.ceil(len(indices) / batch_size),
            position=local_rank,
            dynamic_ncols=True,
        )):
            wav_batch = wav_batch.to(device, non_blocking=True)
            bs = wav_batch.shape[0]
            row_ids = indices[cursor:cursor + bs]
            cursor += bs

            text_tokens,   n_txt_trunc = _tokenize_text_batch(text_tok, texts, text_seq_len, pad_token_text)
            speaker_tokens              = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len)
            speech_tokens, n_sp_trunc  = _tokenize_speech_batch(
                stable_codec, wav_batch, true_wav_lengths, speech_seq_len, pad_token_speech,
            )

            n_text_truncated   += n_txt_trunc
            n_speech_truncated += n_sp_trunc

            combined = torch.cat(
                [text_tokens.to(device), speaker_tokens, speech_tokens], dim=1,
            ).cpu().numpy().astype(np.uint32)  # (B, seq_len)

            arr[row_ids] = combined
            done[row_ids] = 1

            if (batch_idx + 1) % checkpoint_every == 0:
                _checkpoint()

        _checkpoint()
    except Exception:
        _checkpoint()
        raise
    finally:
        del arr, done


# -----------------------------------------------------------------------------
# Cache builder
# -----------------------------------------------------------------------------
def build_packed_cache(
    *,
    hf_path: str,
    hf_config: str,
    hf_split: str,
    split_name: str,
    cache_path: Path,
    meta_path: Path,
    text_tokenizer_name: str,
    text_seq_len: int,
    speaker_model_dir: str,
    speaker_seq_len: int,
    speech_model: str,
    speech_bottleneck: str | None,
    speech_bottleneck_dims: list[int] | None,
    speech_seq_len: int,
    batch_size: int,
    num_workers: int,
    gpu_ids: list[int | None],
    valid_ids: set[str] | None,
    max_duration: float | None,
    text_field: str,
    force: bool,
    checkpoint_every: int = 50,
) -> None:
    # Compute vocabulary and sequence size
    speech_vocab = compute_speech_vocab(speech_bottleneck, speech_bottleneck_dims)
    pad_token_text   = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1
    seq_len = text_seq_len + speaker_seq_len + speech_seq_len

    # Check for complete cache
    if cache_path.exists() and meta_path.exists() and not force:
        print(f'[textaudio-cache] exists -> {cache_path}  (skip; pass --force to rebuild)')
        return

    if total_vocab > 2**32 - 1:
        raise RuntimeError(f'total_vocab={total_vocab} too large for uint32 memmap')

    # Load appropriate dataset and filter valid samples (duration)
    print(f'[textaudio-cache] loading {hf_path!r} config={hf_config!r} split={hf_split!r}')
    dataset = load_dataset(
        hf_path, hf_config, split=hf_split,
    ).cast_column('audio', Audio(decode=False))

    if valid_ids is not None:
        before = len(dataset)
        dataset = dataset.filter(lambda x: x['id'] in valid_ids)
        print(f'[textaudio-cache] duration filter (<= {max_duration}s): '
              f'{before:,} -> {len(dataset):,} samples kept')

    n_samples = len(dataset)
    print(f'[textaudio-cache] {n_samples:,} samples  seq_len={seq_len} -> {cache_path.name}')

    # set up memmap + per-row done tracking (atomic finalize: .tmp -> cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path   = cache_path.with_suffix('.tmp')
    done_path  = cache_path.with_suffix('.done')
    trunc_path = cache_path.with_suffix('.trunc.json')
    stats_dir  = cache_path.parent / f'.{cache_path.stem}_stats'

    fresh = force or not (tmp_path.exists() and done_path.exists())
    if not fresh and done_path.stat().st_size != n_samples:
        print(f'[textaudio-cache] stale checkpoint (n_samples mismatch), starting fresh')
        fresh = True

    if fresh:
        for p in (tmp_path, done_path, trunc_path):
            p.unlink(missing_ok=True)
        arr  = np.memmap(tmp_path,  dtype=np.uint32, mode='w+', shape=(n_samples, seq_len))
        done = np.memmap(done_path, dtype=np.uint8,  mode='w+', shape=(n_samples,))
        del arr, done
        cum_text_trunc = 0
        cum_speech_trunc = 0
    else:
        done = np.memmap(done_path, dtype=np.uint8, mode='r', shape=(n_samples,))
        n_done = int(done.sum())
        print(f'[textaudio-cache] resuming: {n_done:,}/{n_samples:,} samples already done')
        del done
        prev = json.loads(trunc_path.read_text()) if trunc_path.exists() else {}
        cum_text_trunc   = prev.get('n_text_truncated', 0)
        cum_speech_trunc = prev.get('n_speech_truncated', 0)

    done_arr = np.memmap(done_path, dtype=np.uint8, mode='r', shape=(n_samples,))
    undone = np.nonzero(done_arr == 0)[0]
    del done_arr

    if len(undone) == 0:
        os.replace(tmp_path, cache_path)
        done_path.unlink(missing_ok=True)
        trunc_path.unlink(missing_ok=True)
        _write_meta(
            meta_path, hf_path, hf_config, hf_split, split_name, n_samples, seq_len,
            total_vocab, pad_token_text, pad_token_speech, text_seq_len, speaker_seq_len,
            speech_seq_len, speech_vocab, cum_text_trunc, cum_speech_trunc, max_duration,
        )
        print(f'[textaudio-cache] wrote {n_samples:,} × {seq_len} -> {cache_path}')
        return

    # split remaining rows across available workers
    nprocs = min(len(gpu_ids), len(undone))
    index_chunks = [c for c in np.array_split(undone, nprocs) if len(c) > 0]
    nprocs = len(index_chunks)
    active_gpu_ids = gpu_ids[:nprocs]

    stats_dir.mkdir(parents=True, exist_ok=True)

    worker_args = (
        active_gpu_ids, index_chunks, dataset, text_field,
        text_tokenizer_name, speaker_model_dir, speech_model,
        speech_bottleneck, speech_bottleneck_dims,
        tmp_path, done_path, n_samples, seq_len,
        text_seq_len, speaker_seq_len, speech_seq_len,
        pad_token_text, pad_token_speech,
        batch_size, num_workers, checkpoint_every, stats_dir, split_name,
    )

    print(f'[textaudio-cache] {len(undone):,} rows remaining, '
          f'dispatching across {nprocs} worker(s): {active_gpu_ids}')

    try:
        if nprocs == 1:
            _worker_main(0, *worker_args)
        else:
            mp.spawn(_worker_main, args=worker_args, nprocs=nprocs, join=True)
    except Exception:
        _merge_stats(stats_dir, cum_text_trunc, cum_speech_trunc, trunc_path)
        raise

    cum_text_trunc, cum_speech_trunc = _merge_stats(stats_dir, cum_text_trunc, cum_speech_trunc, trunc_path)

    done_arr = np.memmap(done_path, dtype=np.uint8, mode='r', shape=(n_samples,))
    all_done = bool(done_arr.all())
    del done_arr
    if not all_done:
        raise RuntimeError(
            'workers finished but not every row is marked done; rerun to continue.'
        )

    os.replace(tmp_path, cache_path)
    done_path.unlink(missing_ok=True)
    trunc_path.unlink(missing_ok=True)

    _write_meta(
        meta_path, hf_path, hf_config, hf_split, split_name, n_samples, seq_len,
        total_vocab, pad_token_text, pad_token_speech, text_seq_len, speaker_seq_len,
        speech_seq_len, speech_vocab, cum_text_trunc, cum_speech_trunc, max_duration,
    )

    print(f'[textaudio-cache] wrote {n_samples:,} × {seq_len} -> {cache_path}')
    print(f'[textaudio-cache] text_truncated={cum_text_trunc:,}  '
          f'speech_truncated={cum_speech_trunc:,}')
    print(f'[textaudio-cache] meta -> {meta_path}')

# -----------------------------------------------------------------------------
# Write metadata file
# -----------------------------------------------------------------------------

def _write_meta(
    meta_path: Path, hf_path: str, hf_config: str, hf_split: str, split_name: str,
    n_samples: int, seq_len: int, total_vocab: int, pad_token_text: int, pad_token_speech: int,
    text_seq_len: int, speaker_seq_len: int, speech_seq_len: int, speech_vocab: int,
    n_text_truncated: int, n_speech_truncated: int, max_duration: float | None,
) -> None:
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

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build fixed-length multimodal token caches for LibriTTS.',
    )
    # ── LibriTTS (train/val) ─────────────────────────────────────────────────
    ap.add_argument('--splits', type=str, default='libri-train+val+test-clean+test-other')
    ap.add_argument('--hf_path',      type=str, default='mythicinfinity/libritts')
    ap.add_argument('--hf_config',    type=str, default='all')
    ap.add_argument('--train_split',  type=str,
                    default='train.clean.100+train.clean.360+train.other.500')
    ap.add_argument('--train_duration_file', type=str, default='assets/durations/train.csv',
                    help='CSV file (id, duration) for the training split.')
    ap.add_argument('--val_split', type=str, default='dev.clean')
    ap.add_argument('--val_duration_file', type=str, default='assets/durations/val.csv')
    ap.add_argument('--val_size', type=int, default=512)
    # ── LibriSpeech (test) ───────────────────────────────────────────────
    ap.add_argument('--test_hf_path',   type=str,
                    default='mythicinfinity/librispeech-pc-44khz-opus')
    ap.add_argument('--test_clean_config', type=str, default='clean')
    ap.add_argument('--test_clean_split',  type=str, default='test')
    ap.add_argument('--test_other_config', type=str, default='other')
    ap.add_argument('--test_other_split',  type=str, default='test')
    ap.add_argument('--test_clean_duration_file', type=str, default='assets/durations/test_clean.csv',
                    help='CSV file (id, duration) for test.clean.')
    ap.add_argument('--test_other_duration_file', type=str, default='assets/durations/test_other.csv',
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

    ap.add_argument('--batch_size',  type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=32,
                    help='DataLoader worker processes for parallel audio decoding, per GPU.')
    ap.add_argument('--gpus', type=str, default=None,
                    help='Comma-separated CUDA device indices to encode in parallel, '
                         'e.g. "0,1,2,3". Defaults to all visible GPUs, or CPU if none are '
                         'available. The count need not match between runs — progress is '
                         'tracked per-sample and re-split across whatever is available.')
    ap.add_argument('--force',      action='store_true')
    ap.add_argument('--checkpoint_every', type=int, default=25)
    args = ap.parse_args()

    # Speech tokenizer bottleneck default
    if args.speech_bottleneck is None and args.speech_bottleneck_dims is None:
        args.speech_bottleneck = '1x46656_400bps'

    out_dir = Path(args.out_dir)

    # GPU orchestration
    if args.gpus is not None:
        gpu_ids: list[int | None] = [int(x) for x in args.gpus.split(',') if x != '']
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [None]
    print(f'[textaudio-cache] workers: '
          f'{"GPU " + ",".join(map(str, gpu_ids)) if gpu_ids != [None] else "CPU (single process)"}')

    def _load_ids(duration_file: str | None, max_samples: int = -1) -> set[str] | None:
        if duration_file is None:
            return None
        ids = load_valid_ids(duration_file, args.max_duration, max_samples)
        print(f'[textaudio-cache] {len(ids):,} IDs pass duration filter '
              f'(<= {args.max_duration}s) from {duration_file}')
        return ids

    # Parameters shared across all four splits.
    shared = dict(
        text_tokenizer_name=args.text_tokenizer,
        text_seq_len=args.text_seq_len,
        speaker_model_dir=args.speaker_model_dir,
        speaker_seq_len=args.speaker_seq_len,
        speech_model=args.speech_model,
        speech_bottleneck=args.speech_bottleneck,
        speech_bottleneck_dims=args.speech_bottleneck_dims,
        speech_seq_len=args.speech_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
        max_duration=args.max_duration,
        force=args.force,
        checkpoint_every=args.checkpoint_every,
    )
    
    # Cache building for all the selected datasets

    if 'libri-train' in args.splits:
        train_valid_ids      = _load_ids(args.train_duration_file)
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

    # For validation and test, the full sequence is only computed for 
    if 'val' in args.splits:
        val_valid_ids        = _load_ids(args.val_duration_file, args.val_size)
        build_packed_cache(
            hf_path=args.hf_path,
            hf_config=args.hf_config,
            hf_split=args.val_split,
            split_name='val',
            cache_path=out_dir / 'cache_libri_val_asr.uint32',
            meta_path=out_dir / 'cache_libri_val_asr.meta.json',
            valid_ids=val_valid_ids,
            text_field='text_normalized',
            **shared,
        )

    if 'test-clean' in args.splits:
        test_clean_valid_ids = _load_ids(args.test_clean_duration_file)
        build_packed_cache(
            hf_path=args.test_hf_path,
            hf_config=args.test_clean_config,
            hf_split=args.test_clean_split,
            split_name='test_clean',
            cache_path=out_dir / 'cache_libri_test_clean_asr.uint32',
            meta_path=out_dir / 'cache_libri_test_clean_asr.meta.json',
            valid_ids=test_clean_valid_ids,
            text_field='text',
            **shared,
        )

    if 'test-other' in args.splits:
        test_other_valid_ids = _load_ids(args.test_other_duration_file)
        build_packed_cache(
            hf_path=args.test_hf_path,
            hf_config=args.test_other_config,
            hf_split=args.test_other_split,
            split_name='test_other',
            cache_path=out_dir / 'cache_libri_test_other_asr.uint32',
            meta_path=out_dir / 'cache_libri_test_other_asr.meta.json',
            valid_ids=test_other_valid_ids,
            text_field='text',
            **shared,
        )


if __name__ == '__main__':
    main()
