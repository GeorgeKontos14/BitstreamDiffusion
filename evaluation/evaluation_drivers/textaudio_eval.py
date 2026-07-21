
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.textaudio_utils import (
    _read_wav,
    TextAudioEvaluator,
    SALMONEvaluator,
    SALMON_JUDGE_PER_TASK,
)

# -----------------------------------------------------------------------------
# Debug
# -----------------------------------------------------------------------------
def _dbg(msg: str) -> None:
    print(f"[TextAudioEval] {msg}", flush=True)

# -----------------------------------------------------------------------------
# Reading generated samples off disk
# -----------------------------------------------------------------------------
def _load_samples(task_dir: Path) -> List[Dict[str, Any]]:
    path = task_dir / 'samples.json'
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_wavs(run_dir: Path, samples: List[Dict[str, Any]], task_dir: Path, key: str) -> List[Any]:
    wavs = []
    for s in samples:
        rel = s.get(key)
        wavs.append(_read_wav(task_dir / rel) if rel else None)
    return wavs

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def evaluate_textaudio_from_disk(cfg, run_dir: Path, device) -> Dict[str, Dict[str, Dict[str, float]]]:
    run_dir = Path(run_dir)
    manifest_path = run_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run textaudio_generate (run_eval.py --metrics textaudio_generate) first."
        )
    with open(manifest_path, 'r', encoding='utf-8') as f:
        header = json.load(f)['header']

    eval_cfg = getattr(cfg, 'evaluation', None)
    textaudio_cfg = getattr(eval_cfg, 'multimodal_text_audio', None)
    if textaudio_cfg is None:
        raise ValueError("cfg.evaluation.multimodal_text_audio missing; cannot resolve evaluator settings.")

    data_cfg = getattr(cfg, "data", None)
    sample_rate = int(header.get('sample_rate', getattr(data_cfg, "sample_rate", 16000)))
    partition = str(getattr(data_cfg, "partition", "clean"))

    whisper_model = str(getattr(textaudio_cfg, "whisper_model", "openai/whisper-large"))
    speaker_extractor = str(getattr(textaudio_cfg, "speaker_extractor", "microsoft/wavlm-base-plus-sv"))
    speech_extractor = str(getattr(textaudio_cfg, "speech_extractor", "microsoft/wavlm-large"))
    text_model = str(getattr(textaudio_cfg, "text_model", "meta-llama/Llama-3.2-1B"))
    fsd_statistics = getattr(textaudio_cfg, "statistics_path", None)
    spksim_references = getattr(textaudio_cfg, "speaker_ref_path", None)
    asr_num_workers = int(getattr(textaudio_cfg, "asr_num_workers", 4))

    salmon_cfg = getattr(textaudio_cfg, "salmon", None)
    salmon_max_len = float(getattr(salmon_cfg, "max_len", 5.0))
    salmon_ref_dir = str(getattr(salmon_cfg, "ref_dir", "datasets/continuation"))

    tasks: List[str] = list(header.get('tasks', []))
    stt_partitions: List[str] = list(header.get('stt_partitions', ['clean', 'other']))
    sampler_tags: List[str] = [s['tag'] for s in header.get('samplers', [])]

    _dbg(f"run_dir: {run_dir}")
    _dbg(f"Tasks: {tasks}")
    _dbg(f"Samplers: {sampler_tags}")

    evaluator = TextAudioEvaluator(
        whisper_model=whisper_model,
        sr=sample_rate,
        speaker_extractor=speaker_extractor,
        speech_extractor=speech_extractor,
        text_model=text_model,
        statistics_path=fsd_statistics,
        partition=partition,
        _dbg_func=_dbg,
        speaker_ref_path=spksim_references or 'datasets/tts/ref_speaker_embeddings.npz',
        asr_num_workers=asr_num_workers,
    )

    salmon_evaluator: Optional[SALMONEvaluator] = None
    if 'salmon' in tasks:
        salmon_evaluator = SALMONEvaluator(
            sr=sample_rate, max_len=salmon_max_len, ref_dir=salmon_ref_dir, _dbg_func=_dbg,
        )

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {tag: {} for tag in sampler_tags}

    for tag in sampler_tags:
        for task in tasks:
            if task == 'salmon':
                gen_wavs_per_task: Dict[str, List[torch.Tensor]] = {}
                for salmon_task in SALMON_JUDGE_PER_TASK.keys():
                    task_dir = run_dir / tag / 'salmon' / salmon_task
                    samples = _load_samples(task_dir)
                    wavs = _load_wavs(run_dir, samples, task_dir, 'gen_wav')
                    gen_wavs_per_task[salmon_task] = [torch.from_numpy(w) for w in wavs if w is not None]

                n = sum(len(v) for v in gen_wavs_per_task.values())
                if n == 0:
                    _dbg(f"[salmon][{tag}] no samples found on disk; skipping.")
                    continue
                _dbg(f"[salmon][{tag}] evaluating {n} samples ...")
                metrics = salmon_evaluator.evaluate_SALMON(gen_wavs_per_task, device)
                all_results[tag]['salmon'] = metrics
                for name, value in metrics.items():
                    _dbg(f"  salmon/{tag}/{name} = {value:.4f}")
                continue

            if task == 'stt':
                metrics: Dict[str, float] = {}
                for p in stt_partitions:
                    task_dir = run_dir / tag / f'stt_{p}'
                    samples = _load_samples(task_dir)
                    if not samples:
                        _dbg(f"[stt/{p}][{tag}] no samples found on disk; skipping.")
                        continue
                    gen_texts = [s['gen_text'] for s in samples]
                    ref_texts = [s['ref_text'] for s in samples]

                    _dbg(f"[stt/{p}][{tag}] evaluating {len(samples)} samples ...")
                    orig_partition = evaluator.partition
                    evaluator.partition = p
                    try:
                        partition_metrics, _ = evaluator.evaluate_task_extensive(
                            task, gen_texts, ref_texts, [], device,
                        )
                    finally:
                        evaluator.partition = orig_partition
                    metrics.update(partition_metrics)

                all_results[tag]['stt'] = metrics
                for name, value in metrics.items():
                    _dbg(f"  stt/{tag}/{name} = {value:.4f}")
                continue

            # joint / tts / cont
            task_dir = run_dir / tag / task
            samples = _load_samples(task_dir)
            if not samples:
                _dbg(f"[{task}][{tag}] no samples found on disk; skipping.")
                continue

            gen_texts = [s['gen_text'] for s in samples if 'gen_text' in s] or []
            ref_texts = [s['ref_text'] for s in samples if 'ref_text' in s] or []
            gen_wavs = _load_wavs(run_dir, samples, task_dir, 'gen_wav') if task in ('joint', 'tts', 'cont') else []

            _dbg(f"[{task}][{tag}] evaluating {len(samples)} samples ...")
            metrics, _transcriptions = evaluator.evaluate_task_extensive(
                task, gen_texts, ref_texts, gen_wavs, device,
            )
            all_results[tag][task] = metrics
            for name, value in metrics.items():
                _dbg(f"  {task}/{tag}/{name} = {value:.4f}")

    return all_results


def main() -> None:
    import argparse

    from evaluation.evaluation_drivers.utils import _resolve_eval_dirs
    from evaluation.utils import load_config

    ap = argparse.ArgumentParser("Evaluate pre-generated text-audio samples (see textaudio_generate.py)")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--run_dir", type=str, default=None, help="Override the auto-resolved shard directory")
    ap.add_argument("--device", type=str, default=None, help="Override cfg.device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        out_dir, _, _ = _resolve_eval_dirs(cfg)
        ckpt_tag = Path(str(getattr(cfg.evaluation, "checkpoint_path", "checkpoint"))).stem
        run_dir = out_dir / "textaudio_eval" / ckpt_tag

    results = evaluate_textaudio_from_disk(cfg, run_dir, device)
    print(json.dumps(results, indent=2, default=lambda x: None))


if __name__ == "__main__":
    main()
