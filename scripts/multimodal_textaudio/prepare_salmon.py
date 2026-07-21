from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Audio

from cache_utils import (
    MODEL_SR, SPEAKER_OFFSET, SPEAKER_VOCAB, SPEECH_OFFSET,
    load_tokenizers, decode_wav, _tokenize_speaker_batch, _tokenize_speech_batch,
    _pad_wavs, _chunks,
)
from utils.judge_models import JudgeModel

HF_PATH = 'SpeechPPL/SALMon_with_meta'

SALMON_JUDGE_PER_TASK = {
    'sentiment_consistency': 'nvidia/speakerverification_en_titanet_large',
    'speaker_consistency': 'nvidia/speakerverification_en_titanet_large',
    'gender_consistency': 'nvidia/speakerverification_en_titanet_large',
    'bg_domain_consistency': 'ALM/hubert-large-audioset',
    'bg_all_consistency': 'ALM/hubert-large-audioset',
    'rir_consistency': 'ALM/wav2vec2-large-audioset',
}

OUT_DIR = Path('datasets/salmon')

# -----------------------------------------------------------------------------
# Prompt cache: speaker(fixed) + speech(variable, true length only) -- no text
# -----------------------------------------------------------------------------

def build_prompt_cache(
    dataset, task: str, out_dir: Path, speaker_model_dir: str, speech_model: str,
    speaker_seq_len: int, device, batch_size: int,
) -> None:
    _, bicodec, stable_codec, speech_vocab = load_tokenizers(
        text_tokenizer_name='o200k_base', speaker_model_dir=speaker_model_dir,
        speech_model=speech_model, device=device,
    )
    ds_ratio = stable_codec.model.downsampling_ratio
    pad_token_speech = SPEECH_OFFSET + speech_vocab + 1

    n = len(dataset)
    all_tokens: list = [None] * n
    for batch_idx_list in _chunks(list(range(n)), batch_size):
        rows = [dataset[j] for j in batch_idx_list]
        wavs = [decode_wav(r['prompt_audio']['bytes']) for r in rows]
        wav_batch, true_lengths = _pad_wavs(wavs, ds_ratio, device)

        speaker_tokens = _tokenize_speaker_batch(bicodec, wav_batch, speaker_seq_len).cpu()
        max_speech_len = max(math.ceil(l / ds_ratio) for l in true_lengths)
        speech_tokens, _ = _tokenize_speech_batch(
            stable_codec, wav_batch, true_lengths, max_speech_len, pad_token_speech,
        )
        speech_tokens = speech_tokens.cpu()

        for k, j in enumerate(batch_idx_list):
            true_token_len = math.ceil(true_lengths[k] / ds_ratio)
            seq = torch.cat([speaker_tokens[k], speech_tokens[k, :true_token_len]])
            all_tokens[j] = seq.to(torch.int64)
        print(f'[salmon] {task}: prompt-tokenized {min(batch_idx_list[-1] + 1, n)}/{n}', flush=True)

    out_path = out_dir / f'cache_salmon_{task}_prompt.pt'
    torch.save({
        'tokens': all_tokens,
        'n_sequences': n,
        'partition': task,
        'speaker_seq_len': speaker_seq_len,
        'speaker_offset': SPEAKER_OFFSET,
        'speaker_vocab': SPEAKER_VOCAB,
        'speech_offset': SPEECH_OFFSET,
        'speech_vocab': speech_vocab,
    }, out_path)
    print(f'[salmon] {task}: wrote {out_path} ({n} rows)')

# -----------------------------------------------------------------------------
# Judge embeddings: pos/neg continuation audio, embedded by the task's judge
# -----------------------------------------------------------------------------

def build_judge_embeddings(dataset, task: str, out_dir: Path, device, batch_size: int) -> None:
    judge_id = SALMON_JUDGE_PER_TASK[task]
    judge = JudgeModel(judge_id, MODEL_SR, device)

    def _embed_column(column: str) -> np.ndarray:
        embs = []
        for start in range(0, len(dataset), batch_size):
            chunk = dataset[start:start + batch_size][column]
            wavs = [torch.from_numpy(decode_wav(a['bytes'])) for a in chunk]
            embs.append(judge.embed_batch(wavs).detach().cpu())
            print(f'[salmon] {task}: {column} embedded '
                  f'{min(start + batch_size, len(dataset))}/{len(dataset)}', flush=True)
        return torch.cat(embs, dim=0).numpy()

    pos = _embed_column('continuation_audio_positive')
    neg = _embed_column('continuation_audio_negative')
    out_path = out_dir / f'judge_embeddings_{task}.npz'
    np.savez(out_path, positive=pos, negative=neg)
    print(f'[salmon] {task}: wrote {out_path}')

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--speaker_model_dir', type=str,
                     default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speech_model', type=str, default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--speaker_seq_len', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--out_dir', type=str, default=str(OUT_DIR))
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tasks', type=str, nargs='*', default=list(SALMON_JUDGE_PER_TASK.keys()))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    for task in args.tasks:
        print(f'[salmon] loading {HF_PATH!r} config={task!r}')
        dataset = load_dataset(HF_PATH, task, split='train')
        for col in ('prompt_audio', 'continuation_audio_positive', 'continuation_audio_negative'):
            dataset = dataset.cast_column(col, Audio(decode=False))

        build_prompt_cache(
            dataset, task, out_dir, args.speaker_model_dir, args.speech_model,
            args.speaker_seq_len, device, args.batch_size,
        )
        build_judge_embeddings(dataset, task, out_dir, device, args.batch_size)


if __name__ == '__main__':
    main()
