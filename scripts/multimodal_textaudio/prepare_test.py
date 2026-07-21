from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, WavLMForXVector, WavLMModel

from cache_utils import (
    MODEL_SR, load_valid_ids, load_heldout_map, decode_wav,
    build_asr_cache, build_tts_cache, build_continuation_cache,
)

HF_PATH = 'mythicinfinity/librispeech-pc-44khz-opus'
TEXT_FIELD = 'text'

OUT_DIR = Path('datasets/test')
SPEAKER_EMB_OUT = Path('datasets/tts/ref_speaker_embeddings.npz')
FSD_STATS_OUT = Path('assets/libri_test_statistics.npz')
FSD_MODEL = 'microsoft/wavlm-large'
FSD_LAYER = 6

CONT_TRIM_SECONDS = 3.0

# -----------------------------------------------------------------------------
# WavLM x-vector reference speaker embeddings (TTS reference)
# -----------------------------------------------------------------------------

def build_ref_speaker_embeddings(
    dataset, picked: list, partition: str, speaker_extractor_name: str, device, batch_size: int,
) -> None:
    extractor = AutoFeatureExtractor.from_pretrained(speaker_extractor_name)
    model = WavLMForXVector.from_pretrained(speaker_extractor_name).to(device).eval()

    ids = dataset['id']
    kept_ids = [ids[i] for i in picked]
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(picked), batch_size):
            chunk = picked[start:start + batch_size]
            wavs = [decode_wav(dataset[i]['audio']['bytes']) for i in chunk]
            inputs = extractor(wavs, sampling_rate=MODEL_SR, return_tensors='pt', padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embs = model(**inputs).embeddings
            embs = F.normalize(embs, dim=-1).cpu()
            embeddings.append(embs)
            print(f'[test] speaker-embeddings: {min(start + batch_size, len(picked))}/{len(picked)}', flush=True)
    embeddings = torch.cat(embeddings, dim=0).numpy().astype(np.float32)

    SPEAKER_EMB_OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = dict(np.load(SPEAKER_EMB_OUT)) if SPEAKER_EMB_OUT.exists() else {}
    existing[f'{partition}_embeddings'] = embeddings
    existing[f'{partition}_ids'] = np.array(kept_ids)
    np.savez(SPEAKER_EMB_OUT, **existing)
    print(f'[test] wrote {SPEAKER_EMB_OUT} ({partition}_embeddings: {embeddings.shape})')

# -----------------------------------------------------------------------------
# FSD (Frechet Speech Distance) reference statistics
# -----------------------------------------------------------------------------

@torch.no_grad()
def _wavlm_layer_embeddings(wavs: list, extractor, model, device, layer: int, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(wavs), batch_size):
        chunk = wavs[start:start + batch_size]
        inputs = extractor(chunk, sampling_rate=MODEL_SR, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = model(**inputs, output_hidden_states=True).hidden_states[layer]
        out.append(hidden.mean(dim=1).cpu().numpy())
        print(f'[test] fsd: {min(start + batch_size, len(wavs))}/{len(wavs)}', flush=True)
    return np.concatenate(out, axis=0)


def compute_fsd_statistics(hf_config: str, device, batch_size: int) -> tuple:
    dataset = load_dataset(HF_PATH, hf_config, split='test').cast_column('audio', Audio(decode=False))
    wavs = [decode_wav(r['audio']['bytes']) for r in dataset]
    extractor = AutoFeatureExtractor.from_pretrained(FSD_MODEL)
    model = WavLMModel.from_pretrained(FSD_MODEL).to(device).eval()
    embeddings = _wavlm_layer_embeddings(wavs, extractor, model, device, FSD_LAYER, batch_size)
    mean = np.mean(embeddings, axis=0).astype(np.float32)
    cov = np.cov(embeddings, rowvar=False)
    return mean, cov

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_clean_duration_file', type=str, default='assets/durations/test_clean.csv')
    ap.add_argument('--test_other_duration_file', type=str, default='assets/durations/test_other.csv')
    ap.add_argument('--heldout_file', type=str, default='assets/heldout/test_clean.csv')
    ap.add_argument('--max_duration', type=float, default=32.0)

    ap.add_argument('--text_seq_len', type=int, default=100)
    ap.add_argument('--speaker_seq_len', type=int, default=32)
    ap.add_argument('--speech_seq_len', type=int, default=500)

    ap.add_argument('--text_tokenizer', type=str, default='o200k_base')
    ap.add_argument('--speaker_model_dir', type=str,
                     default='Spark-TTS/pretrained_models/SparkTTS-0.5B/BiCodec')
    ap.add_argument('--speech_model', type=str, default='stabilityai/stable-codec-speech-16k')
    ap.add_argument('--speaker_extractor', type=str, default='microsoft/wavlm-base-plus-sv')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_dir', type=str, default=str(OUT_DIR))
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--skip_fsd', action='store_true', help='Skip the FSD statistics pass (slow, whole raw test sets).')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    print(f'[test] loading {HF_PATH!r} config=clean split=test')
    clean_dataset = load_dataset(HF_PATH, 'clean', split='test').cast_column('audio', Audio(decode=False))
    print(f'[test] loading {HF_PATH!r} config=other split=test')
    other_dataset = load_dataset(HF_PATH, 'other', split='test').cast_column('audio', Audio(decode=False))

    heldout_map = load_heldout_map(args.heldout_file)

    # -- test-clean: asr + tts + cont --
    clean_valid_ids = load_valid_ids(args.test_clean_duration_file, args.max_duration)
    build_asr_cache(
        dataset=clean_dataset, out_dir=out_dir, stem='cache_test_clean',
        hf_path=HF_PATH, hf_config='clean', hf_split='test', split_name='test_clean',
        text_field=TEXT_FIELD, valid_ids=clean_valid_ids,
        text_tokenizer_name=args.text_tokenizer, speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        text_seq_len=args.text_seq_len, speaker_seq_len=args.speaker_seq_len,
        speech_seq_len=args.speech_seq_len, device=device, batch_size=args.batch_size,
        size=None, max_duration=args.max_duration,
    )

    tts_picked = build_tts_cache(
        dataset=clean_dataset, out_dir=out_dir, stem='cache_test_clean',
        hf_path=HF_PATH, hf_config='clean', hf_split='test', split_name='test_clean',
        text_field=TEXT_FIELD, valid_ids=clean_valid_ids, heldout_map=heldout_map,
        text_tokenizer_name=args.text_tokenizer, speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        text_seq_len=args.text_seq_len, speaker_seq_len=args.speaker_seq_len,
        speech_seq_len_for_suffix=args.speech_seq_len, device=device, batch_size=args.batch_size,
        size=None,
    )

    cont_valid_ids = load_valid_ids(args.test_clean_duration_file, args.max_duration, min_duration=CONT_TRIM_SECONDS)
    build_continuation_cache(
        dataset=clean_dataset, out_dir=out_dir, stem='cache_test_clean',
        hf_path=HF_PATH, hf_config='clean', hf_split='test', split_name='test_clean',
        valid_ids=cont_valid_ids,
        speaker_model_dir=args.speaker_model_dir, speech_model=args.speech_model,
        speaker_seq_len=args.speaker_seq_len, device=device, batch_size=args.batch_size,
        size=None, trim_seconds=CONT_TRIM_SECONDS, max_duration=args.max_duration,
    )

    # -- test-other: asr only --
    other_valid_ids = load_valid_ids(args.test_other_duration_file, args.max_duration)
    build_asr_cache(
        dataset=other_dataset, out_dir=out_dir, stem='cache_test_other',
        hf_path=HF_PATH, hf_config='other', hf_split='test', split_name='test_other',
        text_field=TEXT_FIELD, valid_ids=other_valid_ids,
        text_tokenizer_name=args.text_tokenizer, speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        text_seq_len=args.text_seq_len, speaker_seq_len=args.speaker_seq_len,
        speech_seq_len=args.speech_seq_len, device=device, batch_size=args.batch_size,
        size=None, max_duration=args.max_duration,
    )

    # -- reference speaker embeddings, aligned to the tts cache's own row selection --
    build_ref_speaker_embeddings(
        clean_dataset, tts_picked, partition='clean',
        speaker_extractor_name=args.speaker_extractor, device=device, batch_size=args.batch_size,
    )

    # -- FSD statistics: raw, unfiltered full test splits --
    if not args.skip_fsd:
        clean_mean, clean_cov = compute_fsd_statistics('clean', device, args.batch_size)
        other_mean, other_cov = compute_fsd_statistics('other', device, args.batch_size)
        FSD_STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            FSD_STATS_OUT,
            clean_mean=clean_mean, clean_cov=clean_cov,
            other_mean=other_mean, other_cov=other_cov,
            layer=FSD_LAYER,
        )
        print(f'[test] wrote {FSD_STATS_OUT}')


if __name__ == '__main__':
    main()
