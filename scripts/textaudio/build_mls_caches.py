from __future__ import annotations

import argparse
import json
import os
import time

from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset, Audio

import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_textaudio_caches import (
    SPEECH_OFFSET,
    compute_speech_vocab,
    _worker_main,
    _merge_stats,
    _write_meta,
)

# -----------------------------------------------------------------------------
# Multi-node logic
# -----------------------------------------------------------------------------

def _node_rank_and_count() -> tuple[int, int]:
    """SLURM_PROCID/SLURM_NTASKS when launched via `srun --ntasks-per-node=1`;
    defaults to a single node when run directly (no SLURM multi-task env)."""
    rank  = int(os.environ.get('SLURM_PROCID', 0))
    count = int(os.environ.get('SLURM_NTASKS', 1))
    return rank, count


def build_mls_cache_multinode(
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
    text_field: str,
    force: bool,
    checkpoint_every: int,
    node_rank: int,
    node_count: int,
    poll_interval: float = 10.0,
) -> None:
    """Same packed-cache format as build_textaudio_caches.build_packed_cache, but
    shards the currently-undone rows across `node_count` SLURM nodes (in addition
    to this node's own GPUs) before each node runs its local mp.spawn worker pool.

    All nodes share the same output files over the cluster's shared filesystem:
    node 0 owns creating/resuming the memmap + done-array (other nodes poll a
    marker file before touching them), and after each node finishes its shard it
    posts its own completion marker; node 0 waits for all of them before doing
    the atomic tmp -> cache_path rename and writing meta.json. No duration
    filtering here (MLS is used as-is), so every node's independent
    `load_dataset` call is guaranteed to produce identical n_samples/ordering.
    """
    speech_vocab = compute_speech_vocab(speech_bottleneck, speech_bottleneck_dims)
    pad_token_text   = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1
    seq_len = text_seq_len + speaker_seq_len + speech_seq_len

    if cache_path.exists() and meta_path.exists() and not force:
        if node_rank == 0:
            print(f'[mls-cache] exists -> {cache_path}  (skip; pass --force to rebuild)')
        return

    if total_vocab > 2**32 - 1:
        raise RuntimeError(f'total_vocab={total_vocab} too large for uint32 memmap')

    print(f'[mls-cache] node {node_rank}/{node_count} loading {hf_path!r} split={hf_split!r}')
    dataset = load_dataset(
        hf_path, hf_config, split=hf_split,
    ).cast_column('audio', Audio(decode=False))
    n_samples = len(dataset)
    print(f'[mls-cache] node {node_rank}: {n_samples:,} samples  seq_len={seq_len} -> {cache_path.name}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path         = cache_path.with_suffix('.tmp')
    done_path        = cache_path.with_suffix('.done')
    setup_ready_path = cache_path.with_suffix('.setup_ready.json')
    stats_dir        = cache_path.parent / f'.{cache_path.stem}_stats_node{node_rank}'
    node_trunc_path  = cache_path.parent / f'.{cache_path.stem}.node{node_rank}.trunc.json'
    node_done_marker = cache_path.parent / f'.{cache_path.stem}.node{node_rank}.complete'

    # ── setup: only node 0 creates/resumes the shared memmap + done array ──
    if node_rank == 0:
        fresh = force or not (tmp_path.exists() and done_path.exists())
        if not fresh and done_path.stat().st_size != n_samples:
            print('[mls-cache] stale checkpoint (n_samples mismatch), starting fresh')
            fresh = True

        if fresh:
            for p in (tmp_path, done_path):
                p.unlink(missing_ok=True)
            for pattern in (f'.{cache_path.stem}.node*.trunc.json', f'.{cache_path.stem}.node*.complete'):
                for p in cache_path.parent.glob(pattern):
                    p.unlink(missing_ok=True)
            arr  = np.memmap(tmp_path,  dtype=np.uint32, mode='w+', shape=(n_samples, seq_len))
            done = np.memmap(done_path, dtype=np.uint8,  mode='w+', shape=(n_samples,))
            del arr, done
        else:
            done = np.memmap(done_path, dtype=np.uint8, mode='r', shape=(n_samples,))
            n_done = int(done.sum())
            print(f'[mls-cache] resuming: {n_done:,}/{n_samples:,} samples already done')
            del done

        setup_ready_path.write_text(json.dumps({'n_samples': n_samples}))
    else:
        print(f'[mls-cache] node {node_rank}: waiting for node 0 to finish setup...')
        while not setup_ready_path.exists():
            time.sleep(poll_interval)
        info = json.loads(setup_ready_path.read_text())
        assert info['n_samples'] == n_samples, (
            f'node {node_rank} sees n_samples={n_samples} but node 0 set up '
            f'{info["n_samples"]!r} -- dataset loading is not reproducing the same rows '
            f'across nodes.'
        )

    # ── shard the currently-undone rows across nodes, then across this node's GPUs ──
    done_arr = np.memmap(done_path, dtype=np.uint8, mode='r', shape=(n_samples,))
    undone = np.nonzero(done_arr == 0)[0]
    del done_arr

    # Sort by duration so each GPU's batches group similar-length clips together.
    durations = np.asarray(dataset['audio_duration'], dtype=np.float64)
    undone = undone[np.argsort(durations[undone])]

    my_chunk = np.array_split(undone, node_count)[node_rank]

    if len(my_chunk) == 0:
        print(f'[mls-cache] node {node_rank}: nothing to do')
    else:
        nprocs = min(len(gpu_ids), len(my_chunk))
        index_chunks = [c for c in np.array_split(my_chunk, nprocs) if len(c) > 0]
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
        print(f'[mls-cache] node {node_rank}: {len(my_chunk):,} rows, '
              f'dispatching across {nprocs} local GPU worker(s): {active_gpu_ids}')

        try:
            if nprocs == 1:
                _worker_main(0, *worker_args)
            else:
                mp.spawn(_worker_main, args=worker_args, nprocs=nprocs, join=True)
        except Exception:
            prev = json.loads(node_trunc_path.read_text()) if node_trunc_path.exists() else {}
            _merge_stats(
                stats_dir, prev.get('n_text_truncated', 0), prev.get('n_speech_truncated', 0),
                node_trunc_path,
            )
            raise

    prev = json.loads(node_trunc_path.read_text()) if node_trunc_path.exists() else {}
    _merge_stats(
        stats_dir, prev.get('n_text_truncated', 0), prev.get('n_speech_truncated', 0),
        node_trunc_path,
    )
    node_done_marker.write_text('done')
    print(f'[mls-cache] node {node_rank}: finished its shard')

    if node_rank != 0:
        return  # only node 0 waits for the others and finalizes

    print(f'[mls-cache] node 0: waiting for all {node_count} node(s) to post completion...')
    marker_paths = [
        cache_path.parent / f'.{cache_path.stem}.node{r}.complete' for r in range(node_count)
    ]
    while not all(p.exists() for p in marker_paths):
        time.sleep(poll_interval)

    cum_text_trunc = 0
    cum_speech_trunc = 0
    for r in range(node_count):
        info = json.loads((cache_path.parent / f'.{cache_path.stem}.node{r}.trunc.json').read_text())
        cum_text_trunc   += info['n_text_truncated']
        cum_speech_trunc += info['n_speech_truncated']

    done_arr = np.memmap(done_path, dtype=np.uint8, mode='r', shape=(n_samples,))
    all_done = bool(done_arr.all())
    del done_arr
    if not all_done:
        raise RuntimeError(
            'all nodes finished but not every row is marked done; rerun to continue.'
        )

    os.replace(tmp_path, cache_path)
    done_path.unlink(missing_ok=True)
    setup_ready_path.unlink(missing_ok=True)
    for r in range(node_count):
        (cache_path.parent / f'.{cache_path.stem}.node{r}.complete').unlink(missing_ok=True)
        (cache_path.parent / f'.{cache_path.stem}.node{r}.trunc.json').unlink(missing_ok=True)

    _write_meta(
        meta_path, hf_path, hf_config, hf_split, split_name, n_samples, seq_len,
        total_vocab, pad_token_text, pad_token_speech, text_seq_len, speaker_seq_len,
        speech_seq_len, speech_vocab, cum_text_trunc, cum_speech_trunc, None,
    )

    print(f'[mls-cache] wrote {n_samples:,} × {seq_len} -> {cache_path}')
    print(f'[mls-cache] text_truncated={cum_text_trunc:,}  speech_truncated={cum_speech_trunc:,}')
    print(f'[mls-cache] meta -> {meta_path}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build a full text+speaker+speech packed token cache (seq_len=1000) '
                     'for the MLS English train split. All samples are already 10-20s and '
                     'all ids are valid, so no duration filtering is applied. Multi-node aware: '
                     'launch via `srun --ntasks-per-node=1` and each node shards the remaining '
                     'work using SLURM_PROCID/SLURM_NTASKS.',
    )
    ap.add_argument('--hf_path',   type=str, default='parler-tts/mls_eng')
    ap.add_argument('--hf_config', type=str, default=None)
    ap.add_argument('--hf_split',  type=str, default='train')

    ap.add_argument('--out_dir', type=str, default='datasets/mls')

    ap.add_argument('--text_tokenizer',    type=str, default='o200k_base')
    ap.add_argument('--text_seq_len',      type=int, default=168)
    ap.add_argument('--speaker_model_dir', type=str,
                    default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speaker_seq_len',   type=int, default=32)
    ap.add_argument('--speech_model',      type=str,
                    default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--speech_seq_len',    type=int, default=800)

    ap.add_argument('--batch_size',  type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=4,
                    help='DataLoader worker processes for parallel audio decoding, per GPU.')
    ap.add_argument('--gpus', type=str, default=None,
                    help='Comma-separated CUDA device indices local to this node, e.g. '
                         '"0,1,2,3". Defaults to all GPUs visible on this node.')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--checkpoint_every', type=int, default=25)
    ap.add_argument('--poll_interval', type=float, default=10.0,
                    help='Seconds between filesystem polls when a node waits on another.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    node_rank, node_count = _node_rank_and_count()

    if args.gpus is not None:
        gpu_ids: list[int | None] = [int(x) for x in args.gpus.split(',') if x != '']
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [None]
    print(f'[mls-cache] node {node_rank}/{node_count}, local workers: '
          f'{"GPU " + ",".join(map(str, gpu_ids)) if gpu_ids != [None] else "CPU (single process)"}')

    build_mls_cache_multinode(
        hf_path=args.hf_path,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        split_name='mls_train',
        cache_path=out_dir / 'cache_mls_train.uint32',
        meta_path=out_dir / 'cache_mls_train.meta.json',
        text_tokenizer_name=args.text_tokenizer,
        text_seq_len=args.text_seq_len,
        speaker_model_dir=args.speaker_model_dir,
        speaker_seq_len=args.speaker_seq_len,
        speech_model=args.speech_model,
        speech_bottleneck='1x46656_400bps',
        speech_bottleneck_dims=None,
        speech_seq_len=args.speech_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu_ids=gpu_ids,
        text_field='transcript',
        force=args.force,
        checkpoint_every=args.checkpoint_every,
        node_rank=node_rank,
        node_count=node_count,
        poll_interval=args.poll_interval,
    )


if __name__ == '__main__':
    main()
