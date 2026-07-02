from __future__ import annotations

import argparse
import io
import json
import math
import os

from pathlib import Path

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
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from build_textaudio_caches import (
    SPEAKER_VOCAB,
    SPEAKER_OFFSET,
    SPEECH_OFFSET,
    MODEL_SR,
    load_valid_ids,
    _tokenize_speaker_batch,
    _tokenize_speech_batch,
)

# -----------------------------------------------------------------------------
# Vocabulary layout
# -----------------------------------------------------------------------------
SPEECH_VOCAB = 46_656

SALMON_CONSISTENCY_SUBSETS = [
    "bg_all_consistency",
    "bg_domain_consistency",
    "gender_consistency",
    "rir_consistency",
    "sentiment_consistency",
    "speaker_consistency",
]

SALMON_TASKS_PER_JUDGE = {
    "nvidia/speakerverification_en_titanet_large": ['sentiment_consistency', 'speaker_consistency', 'gender_consistency'],
    "ALM/hubert-large-audioset": ['bg_domain_consistency', 'bg_all_consistency'],
    "ALM/wav2vec2-large-audioset": ['rir_consistency'],
}

# -----------------------------------------------------------------------------
# Loading tokenizers and .wav files
# -----------------------------------------------------------------------------

def load_speaker_and_speech_tokenizers(
    speaker_model_dir: str,
    speech_model: str,
    device: torch.device,
):
    """Loads only BiCodec + StableCodec -- no text tokenizer, since this script
    never stores text tokens."""
    from sparktts.models.bicodec import BiCodec
    from stable_codec import StableCodec

    print(f'[continuation-cache] loading BiCodec from: {speaker_model_dir}')
    bicodec = BiCodec.load_from_checkpoint(model_dir=speaker_model_dir).to(device).eval()

    print(f'[continuation-cache] loading StableCodec: {speech_model}  bottleneck: 1x46656_400bps')
    stable_codec = StableCodec(pretrained_model=speech_model, device=device)
    stable_codec.set_posthoc_bottleneck('1x46656_400bps')

    return bicodec, stable_codec


def _load_wav(audio_bytes: bytes, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    wav = wav.mean(dim=0)  # mono, shape (T,)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


# -----------------------------------------------------------------------------
# Producing speech and speaker tokens from the prompt:
# first 3 seconds for LibriTTS.dev.clean+LibriSpeech-PC.test.clean
# prompt_audio field for SALMon
# -----------------------------------------------------------------------------

class TrimmedCodecDataset(torch.utils.data.Dataset):
    """Decodes+resamples each row and truncates/pads to exactly `trim_samples`
    samples, so every batch is a plain fixed-size stack -- no dynamic padding
    logic needed."""

    def __init__(self, hf_dataset, target_sr: int, trim_samples: int):
        self.ds = hf_dataset
        self.target_sr = target_sr
        self.trim_samples = trim_samples

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = _load_wav(self.ds[idx]['audio']['bytes'], self.target_sr)
        if wav.shape[-1] < self.trim_samples:
            wav = torch.nn.functional.pad(wav, (0, self.trim_samples - wav.shape[-1]))
        else:
            wav = wav[:self.trim_samples]
        return wav


def _stack_collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch).unsqueeze(1)  # (B, 1, T)


def build_continuation_prefix_cache(
    *,
    hf_path: str,
    hf_config: str,
    hf_split: str,
    split_name: str,
    cache_path: Path,
    meta_path: Path,
    bicodec,
    stable_codec,
    speaker_seq_len: int,
    speech_vocab: int,
    trim_seconds: float,
    device: torch.device,
    valid_ids: set[str] | None,
    min_duration: float,
    max_duration: float,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> None:
    pad_token_text   = SPEECH_OFFSET + speech_vocab
    pad_token_speech = pad_token_text + 1
    total_vocab = pad_token_speech + 1

    ds_ratio = stable_codec.model.downsampling_ratio
    trim_samples = int(round(trim_seconds * MODEL_SR))
    speech_seq_len = trim_samples // ds_ratio
    assert trim_samples % ds_ratio == 0, (
        f'trim_seconds={trim_seconds}s ({trim_samples} samples) is not a multiple of '
        f'downsampling_ratio={ds_ratio}; pick a trim_seconds that divides evenly.'
    )
    seq_len = speaker_seq_len + speech_seq_len

    if cache_path.exists() and meta_path.exists() and not force:
        print(f'[continuation-cache] exists -> {cache_path}  (skip; pass --force to rebuild)')
        return

    print(f'[continuation-cache] loading {hf_path!r} config={hf_config!r} split={hf_split!r}')
    dataset = load_dataset(
        hf_path, hf_config, split=hf_split,
    ).cast_column('audio', Audio(decode=False))

    if valid_ids is not None:
        before = len(dataset)
        dataset = dataset.filter(lambda x: x['id'] in valid_ids)
        print(f'[continuation-cache] duration filter (in [{min_duration}, {max_duration}]s): '
              f'{before:,} -> {len(dataset):,} samples kept')

    n_samples = len(dataset)
    print(f'[continuation-cache] {n_samples:,} samples  seq_len={seq_len} -> {cache_path.name}')

    loader = torch.utils.data.DataLoader(
        TrimmedCodecDataset(dataset, MODEL_SR, trim_samples),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_stack_collate,
        pin_memory=(device.type == 'cuda'),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp')
    arr = np.memmap(tmp_path, dtype=np.uint32, mode='w+', shape=(n_samples, seq_len))

    row = 0
    for wav_batch in tqdm(loader, desc=f'Encoding {split_name}', dynamic_ncols=True,
                           total=math.ceil(n_samples / batch_size)):
        wav_batch = wav_batch.to(device, non_blocking=True)
        bs = wav_batch.shape[0]

        speaker_tokens = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len)
        speech_tokens, _ = _tokenize_speech_batch(
            stable_codec, wav_batch, [trim_samples] * bs, speech_seq_len, pad_token_speech,
        )

        combined = torch.cat([speaker_tokens, speech_tokens], dim=1).cpu().numpy().astype(np.uint32)
        arr[row:row + bs] = combined
        row += bs

    arr.flush()
    del arr
    os.replace(tmp_path, cache_path)

    meta = {
        'cache_format': 'packed_continuation_prefix_blocks',
        'dtype': 'uint32',
        'hf_path': hf_path,
        'hf_config': hf_config,
        'hf_split': hf_split,
        'split_name': split_name,
        'n_sequences': n_samples,
        'seq_len_tokens': seq_len,
        'total_vocab': total_vocab,
        'pad_token_speech': pad_token_speech,
        'speaker_seq_len': speaker_seq_len,
        'speaker_offset': SPEAKER_OFFSET,
        'speaker_vocab': SPEAKER_VOCAB,
        'speech_seq_len': speech_seq_len,
        'speech_offset': SPEECH_OFFSET,
        'speech_vocab': speech_vocab,
        'trim_seconds': trim_seconds,
        'min_duration': min_duration,
        'max_duration': max_duration,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'[continuation-cache] wrote {n_samples:,} × {seq_len} -> {cache_path}')
    print(f'[continuation-cache] meta -> {meta_path}')


def load_salmon_datasets():
    datasets = {}
    for name in SALMON_CONSISTENCY_SUBSETS:
        print(f'[continuation-cache] loading SALMon subset: {name}')
        d = load_dataset(
            'SpeechPPL/SALMon_with_meta', name, split='train',
        ).select_columns(['prompt_audio', 'continuation_audio_positive', 'continuation_audio_negative'])
        d = d.cast_column('prompt_audio', Audio(decode=False))
        d = d.cast_column('continuation_audio_positive', Audio(decode=False))
        d = d.cast_column('continuation_audio_negative', Audio(decode=False))
        datasets[name] = d
    return datasets


def build_salmon_prompt_cache(
    *,
    name: str,
    dataset,
    bicodec,
    stable_codec,
    speaker_seq_len: int,
    speech_vocab: int,
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    force: bool,
) -> None:
    out_path = out_dir / f'cache_salmon_{name}_prompt.pt'
    if out_path.exists() and not force:
        print(f'[continuation-cache] exists -> {out_path}  (skip; pass --force to rebuild)')
        return

    pad_token_speech = SPEECH_OFFSET + speech_vocab + 1
    ds_ratio = stable_codec.model.downsampling_ratio

    all_tokens: list[torch.Tensor] = []
    for start in tqdm(
        range(0, len(dataset), batch_size),
        desc=f'SALMon prompt {name}',
        total=math.ceil(len(dataset) / batch_size),
        dynamic_ncols=True,
    ):
        chunk = dataset[start:start + batch_size]['prompt_audio']
        wavs = [_load_wav(a['bytes'], MODEL_SR) for a in chunk]
        true_lengths = [w.shape[-1] for w in wavs]
        max_len = math.ceil(max(true_lengths) / ds_ratio) * ds_ratio
        padded = torch.stack([
            torch.nn.functional.pad(w, (0, max_len - w.shape[-1])) for w in wavs
        ]).unsqueeze(1).to(device, non_blocking=True)

        speaker_tok = _tokenize_speaker_batch(bicodec, padded, speaker_seq_len).cpu()
        speech_tok, _ = _tokenize_speech_batch(
            stable_codec, padded, true_lengths, max_len // ds_ratio, pad_token_speech,
        )
        speech_tok = speech_tok.cpu()

        for i, true_len in enumerate(true_lengths):
            true_token_len = math.ceil(true_len / ds_ratio)
            seq = torch.cat([speaker_tok[i], speech_tok[i, :true_token_len]])
            all_tokens.append(seq.to(torch.int64))

    torch.save({
        'tokens': all_tokens,
        'n_sequences': len(all_tokens),
        'partition': name,
        'speaker_seq_len': speaker_seq_len,
        'speaker_offset': SPEAKER_OFFSET,
        'speaker_vocab': SPEAKER_VOCAB,
        'speech_offset': SPEECH_OFFSET,
        'speech_vocab': speech_vocab,
    }, out_path)
    lengths = [len(t) for t in all_tokens]
    print(f'[continuation-cache] wrote {len(all_tokens):,} sequences '
          f'(len {min(lengths)}-{max(lengths)}) -> {out_path}')


# -----------------------------------------------------------------------------
# Judge embeddings for SALMon continuations
# -----------------------------------------------------------------------------

def _embed_column(judge, dataset, column: str, batch_size: int, target_sr: int) -> torch.Tensor:
    embs = []
    for start in range(0, len(dataset), batch_size):
        chunk = dataset[start:start + batch_size][column]
        wavs = [_load_wav(a['bytes'], target_sr) for a in chunk]
        embs.append(judge.embed_batch(wavs).detach().cpu())
    return torch.cat(embs, dim=0)


def build_judge_embeddings(
    *,
    datasets: dict,
    tasks_per_judge: dict[str, list[str]],
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    force: bool,
) -> None:
    from utils.judge_models import JudgeModel  # heavy (nemo) import, deferred until actually needed

    for judge_id, tasks in tasks_per_judge.items():
        pending = [
            t for t in tasks
            if force or not (out_dir / f'judge_embeddings_{t}.npz').exists()
        ]
        if not pending:
            print(f'[continuation-cache] judge {judge_id}: all tasks cached, skipping load')
            continue

        print(f'[continuation-cache] loading judge: {judge_id}')
        judge = JudgeModel(judge_id, MODEL_SR, device)

        for task in pending:
            dataset = datasets[task]
            pos = _embed_column(judge, dataset, 'continuation_audio_positive', batch_size, MODEL_SR)
            neg = _embed_column(judge, dataset, 'continuation_audio_negative', batch_size, MODEL_SR)
            out_path = out_dir / f'judge_embeddings_{task}.npz'
            np.savez(out_path, positive=pos.numpy(), negative=neg.numpy())
            print(f'[continuation-cache] wrote {out_path}  '
                  f'positive={tuple(pos.shape)} negative={tuple(neg.shape)}')

        del judge
        if device.type == 'cuda':
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build speaker+speech continuation-prefix caches (3s prompt, no text) '
                     'from LibriTTS dev.clean / LibriSpeech-PC test.clean, then build SALMon '
                     'prompt caches and judge embeddings for the six consistency subsets.',
    )
    ap.add_argument('--libritts_hf_path',   type=str, default='mythicinfinity/libritts')
    ap.add_argument('--libritts_hf_config', type=str, default='clean')
    ap.add_argument('--libritts_hf_split',  type=str, default='dev.clean')
    ap.add_argument('--libritts_duration_file', type=str, default='assets/durations/val.csv')

    ap.add_argument('--librispeech_hf_path',   type=str, default='mythicinfinity/librispeech-pc-44khz-opus')
    ap.add_argument('--librispeech_hf_config', type=str, default='clean')
    ap.add_argument('--librispeech_hf_split',  type=str, default='test')
    ap.add_argument('--librispeech_duration_file', type=str, default='assets/durations/test_clean.csv')

    ap.add_argument('--out_dir',      type=str, default='datasets/continuation')
    ap.add_argument('--min_duration', type=float, default=3.0)
    ap.add_argument('--max_duration', type=float, default=32.0)
    ap.add_argument('--trim_seconds', type=float, default=3.0,
                    help='Prefix length (seconds) used for speaker+speech tokenization.')

    ap.add_argument('--speaker_model_dir', type=str,
                    default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speaker_seq_len',   type=int, default=32)
    ap.add_argument('--speech_model',      type=str,
                    default='stabilityai/stable-codec-speech-16k')

    ap.add_argument('--batch_size',  type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--salmon_batch_size', type=int, default=128,
                    help='Batch size for SALMon prompt tokenization (prompts can be full-length).')
    ap.add_argument('--judge_batch_size', type=int, default=128,
                    help='Batch size for judge-model embedding of continuations.')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--force', action='store_true')

    ap.add_argument('--skip_prefix_caches', action='store_true',
                    help='Skip the LibriTTS/LibriSpeech 3s-prefix caches.')
    ap.add_argument('--skip_salmon', action='store_true',
                    help='Skip the SALMon prompt caches and judge embeddings.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    device  = torch.device(args.device)

    bicodec, stable_codec = load_speaker_and_speech_tokenizers(
        speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        device=device,
    )

    if not args.skip_prefix_caches:
        libritts_valid_ids = load_valid_ids(
            args.libritts_duration_file, args.max_duration, min_duration=args.min_duration,
        )
        librispeech_valid_ids = load_valid_ids(
            args.librispeech_duration_file, args.max_duration, min_duration=args.min_duration,
        )
        print(f'[continuation-cache] {len(libritts_valid_ids):,} LibriTTS IDs in '
              f'[{args.min_duration}, {args.max_duration}]s')
        print(f'[continuation-cache] {len(librispeech_valid_ids):,} LibriSpeech IDs in '
              f'[{args.min_duration}, {args.max_duration}]s')

        prefix_shared = dict(
            bicodec=bicodec,
            stable_codec=stable_codec,
            speaker_seq_len=args.speaker_seq_len,
            speech_vocab=SPEECH_VOCAB,
            trim_seconds=args.trim_seconds,
            device=device,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
        )

        build_continuation_prefix_cache(
            hf_path=args.libritts_hf_path,
            hf_config=args.libritts_hf_config,
            hf_split=args.libritts_hf_split,
            split_name='libritts_dev_clean',
            cache_path=out_dir / 'cache_val_cont.uint32',
            meta_path=out_dir / 'cache_val_cont.meta.json',
            valid_ids=libritts_valid_ids,
            **prefix_shared,
        )

        build_continuation_prefix_cache(
            hf_path=args.librispeech_hf_path,
            hf_config=args.librispeech_hf_config,
            hf_split=args.librispeech_hf_split,
            split_name='librispeech_test_clean',
            cache_path=out_dir / 'cache_test_clean_cont.uint32',
            meta_path=out_dir / 'cache_test_clean_cont.meta.json',
            valid_ids=librispeech_valid_ids,
            **prefix_shared,
        )

    if not args.skip_salmon:
        salmon_datasets = load_salmon_datasets()

        for name, dataset in salmon_datasets.items():
            build_salmon_prompt_cache(
                name=name,
                dataset=dataset,
                bicodec=bicodec,
                stable_codec=stable_codec,
                speaker_seq_len=args.speaker_seq_len,
                speech_vocab=SPEECH_VOCAB,
                device=device,
                out_dir=out_dir,
                batch_size=args.salmon_batch_size,
                force=args.force,
            )

        build_judge_embeddings(
            datasets=salmon_datasets,
            tasks_per_judge=SALMON_TASKS_PER_JUDGE,
            device=device,
            out_dir=out_dir,
            batch_size=args.judge_batch_size,
            force=args.force,
        )


if __name__ == '__main__':
    main()
