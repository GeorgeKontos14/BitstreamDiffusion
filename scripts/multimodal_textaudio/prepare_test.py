from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, WavLMModel

from cache_utils import (
    MODEL_SR, load_valid_ids, load_heldout_map, decode_wav,
    build_asr_cache, build_tts_cache, build_continuation_cache,
)
from utils.speaker_verification import ECAPA_TDNN_SMALL
from utils.textaudio_utils import TextAudioEvaluator

HF_PATH = 'mythicinfinity/librispeech-pc-44khz-opus'
TEXT_FIELD = 'text'

OUT_DIR = Path('datasets/test')
SPEAKER_EMB_OUT = Path('datasets/test/ref_speaker_embeddings.npz')
CONT_TRANSCRIPTIONS_OUT = Path('datasets/test/cache_test_clean_cont.ref_transcriptions.json')
FSD_STATS_OUT = Path('datasets/test/fsd_ref_stats.npz')
FSD_MODEL = 'microsoft/wavlm-large'
FSD_LAYER = 6

FLOW_SLM_SAMPLES_CSV = Path('assets/flow_slm_test_samples.csv')
FSD500_STATS_OUT = Path('datasets/test/fsd_ref_stats_500.npz')
FSD500_E2V_STATS_OUT = Path('datasets/test/fsd_ref_stats_500_e2v.npz')
E2V_MODEL = 'iic/emotion2vec_base'

CONT_TRIM_SECONDS = 3.0

# -----------------------------------------------------------------------------
# WavLM x-vector reference speaker embeddings (TTS reference)
# -----------------------------------------------------------------------------

def build_ref_speaker_embeddings(
    dataset, picked: list, partition: str, speaker_checkpoint_path: str, device,
) -> None:
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    state_dict = torch.load(speaker_checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=False)
    model.eval()
    model.to(device)

    ids = dataset['id']
    kept_ids = [ids[i] for i in picked]
    embeddings = []
    with torch.no_grad():
        for n, i in enumerate(picked):
            wav = decode_wav(dataset[i]['audio']['bytes'])
            x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(device)
            emb = model(x)
            embeddings.append(emb.squeeze(0).cpu())
            if (n + 1) % 200 == 0 or (n + 1) == len(picked):
                print(f'[test] speaker-embeddings: {n + 1}/{len(picked)}', flush=True)
    embeddings = F.normalize(torch.stack(embeddings, dim=0), dim=-1).numpy().astype(np.float32)

    SPEAKER_EMB_OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = dict(np.load(SPEAKER_EMB_OUT)) if SPEAKER_EMB_OUT.exists() else {}
    existing[f'{partition}_embeddings'] = embeddings
    existing[f'{partition}_ids'] = np.array(kept_ids)
    np.savez(SPEAKER_EMB_OUT, **existing)
    print(f'[test] wrote {SPEAKER_EMB_OUT} ({partition}_embeddings: {embeddings.shape})')

# -----------------------------------------------------------------------------
# Continuation reference-prefix transcriptions (first CONT_TRIM_SECONDS of
# each cont row - used for LLM-as-a-Judge)
# -----------------------------------------------------------------------------

def build_cont_prefix_transcriptions(
    dataset, picked: list, whisper_model: str, device,
) -> None:
    trim_samples = int(round(CONT_TRIM_SECONDS * MODEL_SR))
    ids = dataset['id']
    wavs = []
    for n, i in enumerate(picked):
        wavs.append(decode_wav(dataset[i]['audio']['bytes'])[:trim_samples])
        if (n + 1) % 200 == 0 or (n + 1) == len(picked):
            print(f'[test] cont-prefix decode: {n + 1}/{len(picked)}', flush=True)

    evaluator = TextAudioEvaluator(whisper_model=whisper_model, sr=MODEL_SR, _dbg_func=print)
    print(f'[test] transcribing {len(wavs)} continuation prefixes ...', flush=True)
    transcriptions = evaluator.transcribe(wavs, device)

    out = [{'id': ids[i], 'transcription': t} for i, t in zip(picked, transcriptions)]
    CONT_TRANSCRIPTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONT_TRANSCRIPTIONS_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[test] wrote {CONT_TRANSCRIPTIONS_OUT} ({len(out)} rows)')

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
        out.append(hidden.reshape(-1, hidden.size(-1)).cpu().float().numpy())
        print(f'[test] fsd: {min(start + batch_size, len(wavs))}/{len(wavs)}', flush=True)
    return np.concatenate(out, axis=0)


def compute_fsd_statistics(clean_dataset, other_dataset, device, batch_size: int) -> tuple:
    wavs = [decode_wav(r['audio']['bytes']) for r in clean_dataset]
    wavs += [decode_wav(r['audio']['bytes']) for r in other_dataset]
    extractor = AutoFeatureExtractor.from_pretrained(FSD_MODEL)
    model = WavLMModel.from_pretrained(FSD_MODEL).to(device).eval()
    embeddings = _wavlm_layer_embeddings(wavs, extractor, model, device, FSD_LAYER, batch_size)
    mean = np.mean(embeddings, axis=0).astype(np.float32)
    cov = np.cov(embeddings, rowvar=False)
    return mean, cov

# -----------------------------------------------------------------------------
# FSD-wlm / FSD-e2v reference statistics restricted to the Flow-SLM test set
# (cont_flowslm)
# -----------------------------------------------------------------------------

def _parse_flow_slm_csv(path: Path) -> list[tuple[str, str]]:
    """[(partition, id), ...] in file order. e.g.
    'test-other/367/130732/367-130732-0033.flac' -> ('other', '367-130732-0033')."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p = row['path']
            partition = 'clean' if p.startswith('test-clean/') else 'other'
            utt_id = Path(p).stem
            rows.append((partition, utt_id))
    return rows


def _select_flow_slm_wavs(clean_dataset, other_dataset, csv_path: Path) -> list:
    wanted = _parse_flow_slm_csv(csv_path)
    datasets_by_partition = {
        'clean': (clean_dataset, {sid: i for i, sid in enumerate(clean_dataset['id'])}),
        'other': (other_dataset, {sid: i for i, sid in enumerate(other_dataset['id'])}),
    }
    wavs, missing = [], []
    for partition, utt_id in wanted:
        ds, id_to_idx = datasets_by_partition[partition]
        idx = id_to_idx.get(utt_id)
        if idx is None:
            missing.append((partition, utt_id))
            continue
        wavs.append(decode_wav(ds[idx]['audio']['bytes']))
    if missing:
        raise RuntimeError(
            f'{len(missing)}/{len(wanted)} ids from {csv_path} not found in the loaded '
            f'test-clean/test-other datasets: {missing[:10]}'
        )
    print(f'[test] {len(wavs)} utterances resolved from {csv_path} (test-clean + test-other combined)')
    return wavs


@torch.no_grad()
def _e2v_frame_embeddings(wavs, model, device, batch_size: int) -> np.ndarray:
    out = []
    n = len(wavs)
    for start in range(0, n, batch_size):
        chunk = wavs[start:start + batch_size]
        lengths = [len(w) for w in chunk]
        max_len = max(lengths)
        batch = torch.zeros(len(chunk), max_len, dtype=torch.float32, device=device)
        pad_mask = torch.ones(len(chunk), max_len, dtype=torch.bool, device=device)  # True = pad
        for i, w in enumerate(chunk):
            t = torch.as_tensor(w, dtype=torch.float32)
            batch[i, :len(t)] = t.to(device)
            pad_mask[i, :len(t)] = False

        feats = model.extract_features(batch, padding_mask=pad_mask)
        hidden = feats['x']                 # (B, T', C)
        frame_mask = feats['padding_mask']  # (B, T') True = pad, or None if nothing was padded

        for i in range(hidden.size(0)):
            valid = hidden[i] if frame_mask is None else hidden[i][~frame_mask[i]]
            out.append(valid.cpu().float().numpy())
        print(f'[test] fsd-e2v-500: {min(start + batch_size, n)}/{n}', flush=True)
    return np.concatenate(out, axis=0)


def compute_flow_slm_fsd_statistics(
    clean_dataset, other_dataset, device, batch_size: int, e2v_model: str = E2V_MODEL,
) -> tuple:
    """Returns ((wlm_mean, wlm_cov), (e2v_mean, e2v_cov)) computed over the
    Flow-SLM-matched 500-sample subset (FLOW_SLM_SAMPLES_CSV)."""
    wavs = _select_flow_slm_wavs(clean_dataset, other_dataset, FLOW_SLM_SAMPLES_CSV)

    extractor = AutoFeatureExtractor.from_pretrained(FSD_MODEL)
    wlm_model = WavLMModel.from_pretrained(FSD_MODEL).to(device).eval()
    wlm_embeddings = _wavlm_layer_embeddings(wavs, extractor, wlm_model, device, FSD_LAYER, batch_size)
    wlm_mean = np.mean(wlm_embeddings, axis=0).astype(np.float32)
    wlm_cov = np.cov(wlm_embeddings, rowvar=False)
    del wlm_model

    from funasr import AutoModel
    am = AutoModel(model=e2v_model, device=str(device))
    e2v_model_obj = am.model.to(device).eval()
    e2v_embeddings = _e2v_frame_embeddings(wavs, e2v_model_obj, device, batch_size)
    e2v_mean = np.mean(e2v_embeddings, axis=0).astype(np.float32)
    e2v_cov = np.cov(e2v_embeddings, rowvar=False)

    return (wlm_mean, wlm_cov), (e2v_mean, e2v_cov)

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
    ap.add_argument('--speaker_checkpoint', type=str, default='assets/wavlm_large_finetune.pth')
    ap.add_argument('--whisper_model', type=str, default='openai/whisper-large')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_dir', type=str, default=str(OUT_DIR))
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--skip_fsd', action='store_true', help='Skip the FSD statistics pass (slow, whole raw test sets).')
    ap.add_argument(
        '--skip_fsd_flow_slm', action='store_true',
        help='Skip the Flow-SLM-matched (500-sample) FSD-wlm/FSD-e2v reference statistics pass, '
             'used for cont_flowslm scoring.',
    )
    ap.add_argument('--e2v_model', type=str, default=E2V_MODEL)
    ap.add_argument(
        '--skip_cont_transcription', action='store_true',
        help='Skip transcribing the continuation reference prefixes.',
    )
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
        size=None, max_duration=args.max_duration, always_suffix=True,
    )

    tts_picked = build_tts_cache(
        dataset=clean_dataset, out_dir=out_dir, stem='cache_test_clean',
        hf_path=HF_PATH, hf_config='clean', hf_split='test', split_name='test_clean',
        text_field=TEXT_FIELD, valid_ids=clean_valid_ids, heldout_map=heldout_map,
        text_tokenizer_name=args.text_tokenizer, speaker_model_dir=args.speaker_model_dir,
        speech_model=args.speech_model,
        text_seq_len=args.text_seq_len, speaker_seq_len=args.speaker_seq_len,
        speech_seq_len_for_suffix=args.speech_seq_len, device=device, batch_size=args.batch_size,
        size=None, always_suffix=True,
    )

    cont_valid_ids = load_valid_ids(args.test_clean_duration_file, args.max_duration, min_duration=CONT_TRIM_SECONDS)
    cont_picked = build_continuation_cache(
        dataset=clean_dataset, out_dir=out_dir, stem='cache_test_clean',
        hf_path=HF_PATH, hf_config='clean', hf_split='test', split_name='test_clean',
        valid_ids=cont_valid_ids,
        speaker_model_dir=args.speaker_model_dir, speech_model=args.speech_model,
        speaker_seq_len=args.speaker_seq_len, device=device, batch_size=args.batch_size,
        size=None, trim_seconds=CONT_TRIM_SECONDS, max_duration=args.max_duration,
    )

    # -- transcriptions of the cont reference prefix (first CONT_TRIM_SECONDS,
    # not the generated continuation) -- reuses the exact row selection
    # build_continuation_cache just used, so it's guaranteed aligned. --
    if not args.skip_cont_transcription:
        build_cont_prefix_transcriptions(
            clean_dataset, cont_picked, whisper_model=args.whisper_model, device=device,
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
        size=None, max_duration=args.max_duration, always_suffix=True,
    )

    # -- reference speaker embeddings, aligned to the tts cache's own row selection --
    build_ref_speaker_embeddings(
        clean_dataset, tts_picked, partition='clean',
        speaker_checkpoint_path=args.speaker_checkpoint, device=device,
    )

    # -- FSD statistics: raw, unfiltered full test splits, clean+other pooled
    # into a single per-frame mean/covariance (not per-utterance mean-pooled,
    # not split by partition) --
    if not args.skip_fsd:
        mean, cov = compute_fsd_statistics(clean_dataset, other_dataset, device, args.batch_size)
        FSD_STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FSD_STATS_OUT, mean=mean, cov=cov, layer=FSD_LAYER)
        print(f'[test] wrote {FSD_STATS_OUT}')

    # -- FSD-wlm/FSD-e2v statistics restricted to the Flow-SLM-matched
    # 500-sample subset (assets/flow_slm_test_samples.csv), used for
    # cont_flowslm scoring instead of the full-test-set stats above. --
    if not args.skip_fsd_flow_slm:
        (wlm_mean, wlm_cov), (e2v_mean, e2v_cov) = compute_flow_slm_fsd_statistics(
            clean_dataset, other_dataset, device, args.batch_size, e2v_model=args.e2v_model,
        )
        FSD500_STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FSD500_STATS_OUT, mean=wlm_mean, cov=wlm_cov, layer=FSD_LAYER)
        print(f'[test] wrote {FSD500_STATS_OUT}')
        np.savez(FSD500_E2V_STATS_OUT, mean=e2v_mean, cov=e2v_cov, layer='final')
        print(f'[test] wrote {FSD500_E2V_STATS_OUT}')


if __name__ == '__main__':
    main()
