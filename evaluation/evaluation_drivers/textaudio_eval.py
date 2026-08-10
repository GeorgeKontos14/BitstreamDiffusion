
import json
import time
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


def _save_transcriptions(task_dir: Path, samples: List[Dict[str, Any]], transcriptions: List[str]) -> None:
    """Merges Whisper transcriptions back into the samples.json textaudio_generate.py
    already wrote (gen_text/gen_wav/ref_text untouched), so they're no longer
    discarded after being computed for WER/CER/GenPPL-speech scoring."""
    for s, t in zip(samples, transcriptions):
        s['whisper'] = t
    with open(task_dir / 'samples.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def _update_tag_results(run_dir: Path, tag: str, task: str, metrics: Dict[str, float]) -> None:
    path = run_dir / tag / 'results.json'
    path.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    if path.exists():
        try:
            results = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass

    results[task] = metrics
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _discover_tag_manifests(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for manifest_path in sorted(run_dir.glob('*/manifest.json')):
        tag = manifest_path.parent.name
        try:
            out[tag] = json.loads(manifest_path.read_text(encoding='utf-8'))['header']
        except Exception as e:
            _dbg(f"warning: failed to read {manifest_path}: {e}")
    return out

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def evaluate_textaudio_from_disk(
    cfg, run_dir: Path, device,
    tags_override: Optional[List[str]] = None,
    tasks_override: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    run_dir = Path(run_dir)
    tag_headers = _discover_tag_manifests(run_dir)
    if not tag_headers:
        raise FileNotFoundError(
            f"No {run_dir}/*/manifest.json found -- run textaudio_generate "
            "(run_eval.py --metrics textaudio_generate) first."
        )

    if tags_override:
        missing = [t for t in tags_override if t not in tag_headers]
        if missing:
            raise ValueError(
                f"--tags {missing} not found under {run_dir} (available: {sorted(tag_headers)})"
            )
        tag_headers = {t: tag_headers[t] for t in tags_override}

    tasks_by_tag: Dict[str, List[str]] = {}
    for tag, h in tag_headers.items():
        t = list(h.get('tasks', []))
        if tasks_override:
            t = [x for x in t if x in tasks_override]
        tasks_by_tag[tag] = t

    sampler_tags: List[str] = list(tag_headers.keys())
    header = next(iter(tag_headers.values()))  # shared fields agree across tags

    eval_cfg = getattr(cfg, 'evaluation', None)
    textaudio_cfg = getattr(eval_cfg, 'multimodal_text_audio', None)
    if textaudio_cfg is None:
        raise ValueError("cfg.evaluation.multimodal_text_audio missing; cannot resolve evaluator settings.")

    data_cfg = getattr(cfg, "data", None)
    sample_rate = int(header.get('sample_rate', getattr(data_cfg, "sample_rate", 16000)))
    partition = str(getattr(data_cfg, "partition", "clean"))

    whisper_model = str(getattr(textaudio_cfg, "whisper_model", "openai/whisper-large"))
    speaker_checkpoint = str(getattr(textaudio_cfg, "speaker_checkpoint", "assets/wavlm_large_finetune.pth"))
    speech_extractor = str(getattr(textaudio_cfg, "speech_extractor", "microsoft/wavlm-large"))
    text_model = str(getattr(textaudio_cfg, "text_model", "meta-llama/Llama-3.2-1B"))
    fsd_statistics = getattr(textaudio_cfg, "statistics_path", None)
    e2v_statistics = str(getattr(
        textaudio_cfg, "e2v_statistics_path", "datasets/test/fsd_ref_stats_500_e2v.npz"
    ))
    spksim_references = getattr(textaudio_cfg, "speaker_ref_path", None)
    asr_num_workers = int(getattr(textaudio_cfg, "asr_num_workers", 4))

    cont_salmon_cfg = getattr(textaudio_cfg, "cont_salmon", None)
    salmon_max_len = float(getattr(cont_salmon_cfg, "max_len", 5.0))
    salmon_ref_dir = str(getattr(cont_salmon_cfg, "ref_dir", "datasets/continuation"))

    all_tasks_used = sorted({t for tasks in tasks_by_tag.values() for t in tasks})

    _dbg(f"run_dir: {run_dir}")
    _dbg(f"Samplers: {sampler_tags}")
    for tag in sampler_tags:
        _dbg(f"  [{tag}] tasks: {tasks_by_tag[tag]}")

    evaluator = TextAudioEvaluator(
        whisper_model=whisper_model,
        sr=sample_rate,
        speaker_checkpoint=speaker_checkpoint,
        speech_extractor=speech_extractor,
        text_model=text_model,
        statistics_path=fsd_statistics,
        e2v_statistics_path=e2v_statistics,
        partition=partition,
        _dbg_func=_dbg,
        speaker_ref_path=spksim_references or 'datasets/test/ref_speaker_embeddings.npz',
        asr_num_workers=asr_num_workers,
    )

    salmon_evaluator: Optional[SALMONEvaluator] = None
    if 'cont_salmon' in all_tasks_used:
        salmon_evaluator = SALMONEvaluator(
            sr=sample_rate, max_len=salmon_max_len, ref_dir=salmon_ref_dir, _dbg_func=_dbg,
        )

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {tag: {} for tag in sampler_tags}

    for tag in sampler_tags:
        stt_partitions: List[str] = list(tag_headers[tag].get('stt_partitions', ['clean', 'other']))
        for task in tasks_by_tag[tag]:
            if task == 'cont_salmon':
                gen_wavs_per_task: Dict[str, List[torch.Tensor]] = {}
                for salmon_task in SALMON_JUDGE_PER_TASK.keys():
                    task_dir = run_dir / tag / 'cont_salmon' / salmon_task
                    samples = _load_samples(task_dir)
                    wavs = _load_wavs(run_dir, samples, task_dir, 'gen_wav')
                    gen_wavs_per_task[salmon_task] = [torch.from_numpy(w) for w in wavs if w is not None]

                n = sum(len(v) for v in gen_wavs_per_task.values())
                if n == 0:
                    _dbg(f"[cont_salmon][{tag}] no samples found on disk; skipping.")
                    continue
                _dbg(f"[cont_salmon][{tag}] evaluating {n} samples ...")
                t0 = time.monotonic()
                metrics = salmon_evaluator.evaluate_SALMON(gen_wavs_per_task, device)
                all_results[tag]['cont_salmon'] = metrics
                _update_tag_results(run_dir, tag, 'cont_salmon', metrics)
                for name, value in metrics.items():
                    _dbg(f"  cont_salmon/{tag}/{name} = {value:.4f}")
                _dbg(f"[cont_salmon][{tag}] done in {time.monotonic() - t0:.1f}s")
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
                    t0 = time.monotonic()
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
                    _update_tag_results(run_dir, tag, 'stt', metrics)
                    _dbg(f"[stt/{p}][{tag}] done in {time.monotonic() - t0:.1f}s")

                for name, value in metrics.items():
                    _dbg(f"  stt/{tag}/{name} = {value:.4f}")
                continue

            # joint / tts / cont_taste / cont_flowslm
            task_dir = run_dir / tag / task
            samples = _load_samples(task_dir)
            if not samples:
                _dbg(f"[{task}][{tag}] no samples found on disk; skipping.")
                continue

            gen_texts = [s['gen_text'] for s in samples if 'gen_text' in s] or []
            ref_texts = [s['ref_text'] for s in samples if 'ref_text' in s] or []
            gen_wavs = (
                _load_wavs(run_dir, samples, task_dir, 'gen_wav')
                if task in ('joint', 'tts', 'cont_taste', 'cont_flowslm') else []
            )

            _dbg(f"[{task}][{tag}] evaluating {len(samples)} samples ...")
            t0 = time.monotonic()
            metrics, transcriptions = evaluator.evaluate_task_extensive(
                task, gen_texts, ref_texts, gen_wavs, device,
            )
            all_results[tag][task] = metrics
            _update_tag_results(run_dir, tag, task, metrics)
            if transcriptions is not None:
                _save_transcriptions(task_dir, samples, transcriptions)
            for name, value in metrics.items():
                _dbg(f"  {task}/{tag}/{name} = {value:.4f}")
            _dbg(f"[{task}][{tag}] done in {time.monotonic() - t0:.1f}s")

    return all_results


def main() -> None:
    import argparse

    from evaluation.evaluation_drivers.utils import _resolve_eval_dirs
    from evaluation.utils import load_config

    ap = argparse.ArgumentParser("Evaluate pre-generated text-audio samples (see textaudio_generate.py)")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--run_dir", type=str, default=None, help="Override the auto-resolved shard directory")
    ap.add_argument("--device", type=str, default=None, help="Override cfg.device")
    ap.add_argument(
        "--tags", nargs="+", default=None,
        help="Only evaluate these sampler tags (run_dir subdirectory names), instead of every "
             "tag with a manifest.json under run_dir. Use this to re-run/extend scoring for one "
             "tag without repeating work already done for others.",
    )
    ap.add_argument(
        "--tasks", nargs="+", default=None,
        help="Only evaluate these tasks (e.g. joint tts stt cont_taste cont_flowslm cont_salmon), "
             "intersected against whatever each selected tag's manifest.json actually has samples for.",
    )
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

    results = evaluate_textaudio_from_disk(
        cfg, run_dir, device, tags_override=args.tags, tasks_override=args.tasks,
    )
    print(json.dumps(results, indent=2, default=lambda x: None))


if __name__ == "__main__":
    main()
