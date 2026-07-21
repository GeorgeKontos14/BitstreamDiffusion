
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast


try:
    import torch.distributed as dist
except Exception:
    dist = None

from data import get_loader
from diffusion.continuous.processes import ContinuousForwardProcess
from evaluation.distributed import barrier, gather_varlen_firstdim_to_rank0
from evaluation.evaluation_drivers.utils import _resolve_eval_dirs
from evaluation.utils import resolve_eval_amp, resolve_entropy_run_dir
from utils.model_utils import unwrap_model
from utils.textaudio_utils import (
    _fixed_mask,
    _safe_decode,
    _write_wav,
    load_ref_audio_cache,
    SALMON_JUDGE_PER_TASK,
    UNCONDITIONAL,
    TEXT_TO_SPEECH,
    SPEECH_TO_TEXT,
    SPEECH_CONTINUATION,
)

# -----------------------------------------------------------------------------
# DDP helpers
# -----------------------------------------------------------------------------
def _ddp_is_on() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()

def _rank() -> int:
    return int(dist.get_rank()) if _ddp_is_on() else 0

def _world() -> int:
    return int(dist.get_world_size()) if _ddp_is_on() else 1


def _rank0() -> bool:
    return (not _ddp_is_on()) or (_rank() == 0)


def _dbg(msg: str, *, all_ranks: bool = False) -> None:
    if all_ranks:
        print(f"[TextAudioGen][rank{_rank()}/{_world()}] {msg}", flush=True)
    elif _rank0():
        print(f"[TextAudioGen][rank{_rank()}/{_world()}] {msg}", flush=True)

# -----------------------------------------------------------------------------
# Task Definition
# -----------------------------------------------------------------------------
TASKS = ['joint', 'tts', 'stt', 'cont', 'salmon']
DEFAULT_TASKS = ['joint', 'tts', 'stt', 'cont']
TASK_IDS = {
    'joint': UNCONDITIONAL,
    'tts': TEXT_TO_SPEECH,
    'stt': SPEECH_TO_TEXT,
    'cont': SPEECH_CONTINUATION,
}

# -----------------------------------------------------------------------------
# Sampler specs
# -----------------------------------------------------------------------------
@dataclass
class _SamplerSpec:
    tag: str
    sampler_name: str
    num_steps: int
    target_nfe: int
    actual_nfe: int
    stochastic_enabled: bool
    s_churn: Optional[float]  # extract from config
    gamma: Optional[float]    # report
    guidance_scale: float

def _build_sampler_specs(cfg: Any, c: Any) -> List[_SamplerSpec]:
    """
    Expands cfg.evaluation.multimodal_text_audio.sampling_sweep.specs into one
    _SamplerSpec per (target_nfe x s_churn x guidance_scale) combination.
    """
    from evaluation.nfe import steps_for_target_nfe

    framework = str(getattr(cfg, 'framework', 'continuous_score'))
    self_condition = bool(getattr(getattr(cfg, 'model', None), 'self_condition', False))

    sweep = getattr(c, 'sampling_sweep', None) if c is not None else None
    raw_specs = list(getattr(sweep, 'specs', [])) if sweep is not None else []
    if not raw_specs:
        raise ValueError(
            "cfg.evaluation.multimodal_text_audio.sampling_sweep.specs must define at least one spec "
            "(sampler_name + target_nfes, optionally stochastic_enabled + s_churns)."
        )

    specs: List[_SamplerSpec] = []
    seen_tags = set()

    for base in raw_specs:
        sampler_name = str(base.sampler_name)
        target_nfes = [int(n) for n in base.target_nfes]
        stochastic_enabled = bool(getattr(base, 'stochastic_enabled', False))

        if stochastic_enabled:
            s_churns = [float(v) for v in getattr(base, 's_churns', [])]
            if not s_churns:
                raise ValueError(
                    f"stochastic_enabled=True but no s_churns given for sampler_name={sampler_name!r}"
                )
        else:
            s_churns = [None]

        guidance_scales = [float(v) for v in getattr(base, 'guidance_scales', [0.0])]
        if not guidance_scales:
            guidance_scales = [0.0]

        for target_nfe in target_nfes:
            num_steps, actual_nfe = steps_for_target_nfe(
                framework=framework,
                sampler_name=sampler_name,
                target_nfe=target_nfe,
                self_condition=self_condition,
                sc_refresh_mode="refined",
                return_probs=True,
            )
            for s_churn in s_churns:
                gamma = (
                    min(float(s_churn) / float(num_steps), math.sqrt(2.0) - 1.0)
                    if (stochastic_enabled and num_steps > 0) else None
                )
                for guidance_scale in guidance_scales:
                    tag = f"{sampler_name}_nfe{target_nfe}"
                    if stochastic_enabled:
                        tag += f"_stoch-g{gamma:.4g}"
                    if guidance_scale > 0.0:
                        tag += f"_gs{guidance_scale:g}"

                    if tag in seen_tags:
                        raise ValueError(f"Duplicate textaudio sampling spec tag: {tag!r}")
                    seen_tags.add(tag)

                    specs.append(_SamplerSpec(
                        tag=tag,
                        sampler_name=sampler_name,
                        num_steps=num_steps,
                        target_nfe=target_nfe,
                        actual_nfe=actual_nfe,
                        stochastic_enabled=stochastic_enabled,
                        s_churn=s_churn,
                        gamma=gamma,
                        guidance_scale=guidance_scale,
                    ))

    return specs

# -----------------------------------------------------------------------------
# Sharding and multi-gpu test-loaders
# -----------------------------------------------------------------------------
def _shard_indices(n: int, world_size: int, rank: int) -> List[int]:
    base, rem = divmod(int(n), int(world_size))
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))

def get_sharded_test_loaders(
    cfg, batch_size: int, stt_partitions: List[str], rank: int, world_size: int,
) -> Dict[str, Any]:
    from torch.utils.data import DataLoader, Subset

    def _sharded(ds):
        idx = _shard_indices(len(ds), world_size, rank)
        return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, drop_last=False)

    loaders: Dict[str, Any] = {
        'tts': _sharded(get_loader(cfg, split='test', task='tts', batch_size=batch_size).dataset),
        'cont': _sharded(get_loader(cfg, split='test', task='cont', batch_size=batch_size).dataset),
    }

    data_cfg = cfg.data
    orig_partition = getattr(data_cfg, 'partition', 'clean')
    stt_loaders: Dict[str, Any] = {}
    try:
        for p in stt_partitions:
            data_cfg.partition = p
            ds = get_loader(cfg, split='test', batch_size=batch_size).dataset
            stt_loaders[p] = _sharded(ds)
    finally:
        data_cfg.partition = orig_partition
    loaders['stt'] = stt_loaders

    return loaders

# -----------------------------------------------------------------------------
# Generation helpers
# -----------------------------------------------------------------------------
def _get_sampler(model, proc, cfg, sampler_name: str, cache: Dict[str, Any]):
    if "ddim" in sampler_name:
        from diffusion.continuous.samplers import DDIMSampler
        if cache.get("ddim") is None:
            cache["ddim"] = DDIMSampler(model, proc, cfg)
        return cache["ddim"]
    if cache.get("heun") is None:
        from diffusion.continuous.samplers import HeunSampler
        cache["heun"] = HeunSampler(model, proc, cfg)
    return cache["heun"]


@contextmanager
def _stochastic_cfg_override(cfg: Any, overrides: Dict[str, Any]):
    if not overrides:
        yield
        return

    ev = getattr(cfg, 'evaluation', None)
    if ev is None:
        yield
        return

    st = getattr(ev, 'stochastic', None)
    st_existed = st is not None
    if st is None:
        from ml_collections import config_dict
        ev.stochastic = config_dict.ConfigDict()
        st = ev.stochastic

    backup = {}
    for key, value in overrides.items():
        backup[key] = (hasattr(st, key), getattr(st, key, None))
        setattr(st, key, value)

    try:
        yield
    finally:
        for key, (existed, old_val) in backup.items():
            if existed:
                setattr(st, key, old_val)
            else:
                try:
                    delattr(st, key)
                except AttributeError:
                    pass
        if not st_existed:
            try:
                delattr(ev, 'stochastic')
            except AttributeError:
                pass


def _generate_with_mask(
    model, proc, cfg, x_full: torch.Tensor, cond_mask: torch.Tensor, B: int,
    sequence_len: int, spec: _SamplerSpec,
    terminal_sigma: float, entropic_blend_alpha: float, entropy_run_dir,
    sampler_cache: Dict[str, Any],
) -> torch.Tensor:
    schedule = 'entropic' if 'entropic' in spec.sampler_name else 'karras'
    sampler_obj = _get_sampler(model, proc, cfg, spec.sampler_name, sampler_cache)

    stoch_overrides: Dict[str, Any] = {}
    if not spec.stochastic_enabled:
        stoch_overrides['enabled'] = False
    else:
        stoch_overrides['enabled'] = True
        if spec.s_churn is not None:
            stoch_overrides['s_churn'] = spec.s_churn

    cond_kwargs = dict(
        conditioning_prefix_full=x_full,
        cond_prefix_mask=cond_mask,
        guidance_scale=spec.guidance_scale,
    )

    with _stochastic_cfg_override(cfg, stoch_overrides):
        _, probs = sampler_obj.sample(
            B, sequence_len,
            schedule=schedule,
            num_steps=spec.num_steps,
            entropic_blend_alpha=entropic_blend_alpha,
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=terminal_sigma,
            return_probs=True,
            progress=False,
            **cond_kwargs,
        )

    bits = (probs > 0.5).to(torch.long)
    bits[cond_mask] = (x_full[cond_mask] > 0.5).to(torch.long)
    return bits


def _generate_batch(
    model, proc, cfg, x_full: torch.Tensor, task: str, B: int,
    sequence_len: int, bits_per_token: int, spec: _SamplerSpec,
    terminal_sigma: float, entropic_blend_alpha: float, entropy_run_dir,
    sampler_cache: Dict[str, Any],
) -> torch.Tensor:
    cond_mask = _fixed_mask(
        cfg, B, sequence_len, TASK_IDS[task], device=x_full.device, bits_per_token=bits_per_token
    )
    return _generate_with_mask(
        model, proc, cfg, x_full, cond_mask, B, sequence_len, spec,
        terminal_sigma, entropic_blend_alpha, entropy_run_dir, sampler_cache,
    )


def _prepare_x_full(
    batch: torch.Tensor, task: str, sequence_len: int, bits_per_token: int,
    text_seq_len: int, device,
) -> torch.Tensor:
    batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
    if task == 'stt':
        return batch
    start = 0 if task == 'tts' else text_seq_len * bits_per_token
    full = torch.zeros(batch.size(0), sequence_len, device=device, dtype=torch.float32)
    full[:, start:start + batch.size(1)] = batch
    return full


def _token_ids_to_bits(token_ids: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    shifts = torch.arange(bits_per_token - 1, -1, -1, device=token_ids.device)
    return ((token_ids.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def _load_salmon_prompts(ref_dir: str) -> Dict[str, List[torch.Tensor]]:
    prompts: Dict[str, List[torch.Tensor]] = {}
    for task in SALMON_JUDGE_PER_TASK.keys():
        path = Path(ref_dir) / f'cache_salmon_{task}_prompt.pt'
        data = torch.load(path, map_location='cpu', weights_only=False)
        prompts[task] = [row.to(torch.long) for row in data['tokens']]
    return prompts


def _salmon_batch_x_full_and_mask(
    rows: List[torch.Tensor], sequence_len: int, bits_per_token: int, text_seq_len: int, device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(rows)
    start = text_seq_len * bits_per_token
    x_full = torch.zeros(B, sequence_len, device=device, dtype=torch.float32)
    cond_mask = torch.zeros(B, sequence_len, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        bits = _token_ids_to_bits(row.to(device=device), bits_per_token).view(-1)
        end = start + bits.numel()
        x_full[i, start:end] = bits
        cond_mask[i, start:end] = True
    return x_full, cond_mask


def _bits_to_token_ids(bits: torch.Tensor, bits_per_token: int) -> torch.Tensor:
    B = bits.size(0)
    tok_bits = bits.view(B, -1, bits_per_token)
    shifts = torch.arange(bits_per_token - 1, -1, -1, device=bits.device)
    return (tok_bits * (2 ** shifts)).sum(dim=-1).long()


def _decode_texts(
    text_tok, token_ids: torch.Tensor, text_seq_len: int, speech_offset: int, speech_vocab_size: int,
) -> List[str]:
    pad_text = speech_offset + speech_vocab_size
    ids = token_ids[:, :text_seq_len]
    pad = ids == pad_text
    ids_clean = ids.clone()
    ids_clean[pad] = 0

    texts = []
    for i in range(ids.size(0)):
        valid = ids_clean[i][~pad[i]].cpu().tolist()
        texts.append(_safe_decode(text_tok, valid))
    return texts


def _decode_speech_wavs(
    speech_tok, token_ids: torch.Tensor, text_seq_len: int, speaker_seq_len: int,
    speech_offset: int, speech_vocab_size: int,
):
    pad_speech = speech_offset + speech_vocab_size + 1
    global_ids = token_ids[:, text_seq_len + speaker_seq_len:]
    pad = global_ids == pad_speech
    local_ids = (global_ids - speech_offset).clamp(0, speech_vocab_size - 1)

    wavs = []
    for i in range(local_ids.size(0)):
        valid = local_ids[i][~pad[i]]
        wav = speech_tok.decode([valid.unsqueeze(0).unsqueeze(-1)], posthoc_bottleneck=True)
        wav = wav.squeeze(0).cpu().float().squeeze(0).numpy()
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = wav / peak
        wavs.append(wav)
    return wavs

# -----------------------------------------------------------------------------
# Gather helper -- concatenates a rank's local batches, then gathers to
# rank0. Safe to call exactly once per (task, sampler[, sub-key]): each rank
# calls it the same number of times regardless of local shard size, so there
# is no per-batch collective-count mismatch hazard.
# -----------------------------------------------------------------------------
def _gather_bits(local_batches: List[torch.Tensor], sequence_len: int, device) -> Optional[torch.Tensor]:
    local = torch.cat(local_batches, dim=0) if local_batches else torch.zeros(0, sequence_len, device=device, dtype=torch.long)
    return gather_varlen_firstdim_to_rank0(local, dst=0)

# -----------------------------------------------------------------------------
# On-disk writer (rank0 only)
# -----------------------------------------------------------------------------
def _write_manifest(run_dir: Path, header: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'header': header}, f, indent=2, ensure_ascii=False)


def _write_samples(task_dir: Path, samples: List[Dict[str, Any]]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / 'samples.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def generate_textaudio(cfg, model, device, rank: int, world_size: int) -> Optional[Path]:
    eval_cfg = getattr(cfg, 'evaluation', None)
    textaudio_cfg = getattr(eval_cfg, 'multimodal_text_audio', None)

    if textaudio_cfg is None:
        _dbg("⚠️  cfg.evaluation.multimodal_text_audio missing; skipping text-audio generation.")
        return None

    seed = int(getattr(textaudio_cfg, 'seed', 42))

    data_cfg = getattr(cfg, "data", None)
    bits_per_token  = int(getattr(data_cfg, "bits_per_token",   18))
    text_seq_len    = int(getattr(data_cfg, "text_seq_len",     168))
    speaker_seq_len = int(getattr(data_cfg, "speaker_seq_len",  32))
    speech_offset   = int(getattr(data_cfg, "speech_offset",    204115))
    speech_vocab_sz = int(getattr(data_cfg, "speech_vocab_size", 46656))
    sequence_len    = int(getattr(data_cfg, "sequence_len",     18000))
    sample_rate     = int(getattr(data_cfg, "sample_rate",      16000))
    data_root       = getattr(data_cfg, "root", "datasets/")

    tasks = [t for t in getattr(textaudio_cfg, "tasks", DEFAULT_TASKS) if t in TASKS]
    stt_partitions = [str(p) for p in getattr(textaudio_cfg, "stt_partitions", ["clean", "other"])]

    salmon_cfg = getattr(textaudio_cfg, "salmon", None)
    salmon_enabled = bool(getattr(salmon_cfg, "enabled", False))
    salmon_ref_dir = str(getattr(salmon_cfg, "ref_dir", "datasets/continuation"))
    if salmon_enabled and "salmon" not in tasks:
        tasks.append("salmon")

    terminal_sigma = float(getattr(textaudio_cfg, "terminal_sigma", 0.08))
    entropic_blend_alpha = float(getattr(textaudio_cfg, "entropic_blend_alpha", 0.0))
    entropy_run_dir = getattr(textaudio_cfg, "entropy_run_dir", None)
    if entropy_run_dir is None:
        entropy_run_dir = resolve_entropy_run_dir(cfg, textaudio_cfg)

    num_uncond_samples = int(getattr(textaudio_cfg, "num_uncond_samples", 1024))
    gen_batch_size = int(
        getattr(textaudio_cfg, "batch_size", getattr(eval_cfg, "batch_size", getattr(cfg.train, "batch_size", 32)))
    )

    samplers = _build_sampler_specs(cfg, textaudio_cfg)

    fw = str(getattr(cfg, 'framework', 'continuous_score'))
    if fw != 'continuous_score':
        raise ValueError(f"textaudio generation currently supports framework='continuous_score' only (got {fw!r}).")

    use_amp, amp_dtype = resolve_eval_amp(device, cfg)

    out_dir, _, _ = _resolve_eval_dirs(cfg)
    ckpt_tag = Path(str(getattr(cfg.evaluation, "checkpoint_path", "checkpoint"))).stem
    run_dir = out_dir / "textaudio_eval" / ckpt_tag

    _dbg(f"Tasks: {tasks}")
    _dbg(f"Samplers: {[spec.tag for spec in samplers]}")
    _dbg(f"world_size={world_size} -> each task's samples sharded across ranks, gathered to rank0 for decode+write")
    _dbg(f"run_dir: {run_dir}")

    if _rank0():
        header = {
            'title': 'Single-stream Joint Audio+Text Generation (multi-GPU generation, rank0-written)',
            'experiment': str(getattr(cfg, "experiment", "unknown")),
            'text_tokenizer': str(getattr(data_cfg, "text_tokenizer", "o200k_base")),
            'speaker_tokenizer': str(getattr(data_cfg, "speaker_tokenizer", "bicodec")),
            'speech_tokenizer': str(getattr(data_cfg, "speech_tokenizer", "stabilityai/stable-codec-speech-16k")),
            'bits_per_token': bits_per_token,
            'sequence_layout': {
                'text_tokens': text_seq_len,
                'speaker_tokens': speaker_seq_len,
                'total_tokens': int(sequence_len // bits_per_token),
            },
            'checkpoint': str(getattr(cfg.evaluation, "checkpoint_path", "")),
            'tasks': tasks,
            'stt_partitions': stt_partitions,
            'num_uncond_samples': num_uncond_samples,
            'samplers': [
                {
                    'tag': s.tag, 'sampler_name': s.sampler_name, 'num_steps': s.num_steps,
                    'target_nfe': s.target_nfe, 'actual_nfe': s.actual_nfe,
                    'stochastic_enabled': s.stochastic_enabled, 'gamma': s.gamma,
                    'guidance_scale': s.guidance_scale,
                }
                for s in samplers
            ],
            'sample_rate': sample_rate,
            'world_size': world_size,
        }
        _write_manifest(run_dir, header)
    barrier()

    # Decoding (text/speech tokenizers) only ever happens on rank0, after the
    # gather -- other ranks only need the diffusion model + sampler to
    # generate bits, so they skip loading these (and the ref-audio caches).
    text_tok = speech_tok = None
    if _rank0():
        from trainers.trainer import _load_text_speech_tokenizers
        text_tok, speech_tok = _load_text_speech_tokenizers(cfg, device)

    proc = ContinuousForwardProcess(cfg)
    raw_model = unwrap_model(model)
    sampler_cache: Dict[str, Any] = {}

    torch.manual_seed(seed + 1000 * rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1000 * rank)

    salmon_prompts = _load_salmon_prompts(salmon_ref_dir) if 'salmon' in tasks else {}

    loaders: Dict[str, Any] = {}
    if any(t in ('tts', 'cont', 'stt') for t in tasks):
        loaders = get_sharded_test_loaders(cfg, gen_batch_size, stt_partitions, rank, world_size)

    tts_ref_cache = (
        load_ref_audio_cache(data_root, split='test', task='tts', partition='clean', dbg=_dbg)
        if (_rank0() and 'tts' in tasks) else None
    )
    stt_ref_cache_by_partition = (
        {p: load_ref_audio_cache(data_root, split='test', task='asr', partition=p, dbg=_dbg) for p in stt_partitions}
        if (_rank0() and 'stt' in tasks) else {}
    )

    model.eval()
    with torch.no_grad():
        for task in tasks:
            for spec_idx, spec in enumerate(samplers):
                write_ref = (spec_idx == 0)

                # Unconditional generation can't be CFG-guided
                if task == 'joint' and spec.guidance_scale > 0.0:
                    _dbg(f"[joint][{spec.tag}] skipping unconditional generation "
                         f"(guidance_scale={spec.guidance_scale} > 0).", all_ranks=True)
                    continue

                _dbg(f"[{task}][{spec.tag}] generating (steps={spec.num_steps}) ...", all_ranks=True)

                # ---------------------------------------------------------------
                # salmon: 6 sub-tasks, variable-length prefixes, wav-only
                # ---------------------------------------------------------------
                if task == 'salmon':
                    for salmon_task, rows in salmon_prompts.items():
                        my_idx = _shard_indices(len(rows), world_size, rank)
                        my_rows = [rows[i] for i in my_idx]
                        local_bits: List[torch.Tensor] = []
                        for i in range(0, len(my_rows), gen_batch_size):
                            chunk = my_rows[i:i + gen_batch_size]
                            x_full, cond_mask = _salmon_batch_x_full_and_mask(
                                chunk, sequence_len, bits_per_token, text_seq_len, device,
                            )
                            B = x_full.size(0)
                            with autocast(enabled=use_amp, dtype=amp_dtype):
                                bits = _generate_with_mask(
                                    raw_model, proc, cfg, x_full, cond_mask, B, sequence_len, spec,
                                    terminal_sigma, entropic_blend_alpha, entropy_run_dir, sampler_cache,
                                )
                            local_bits.append(bits)

                        gathered = _gather_bits(local_bits, sequence_len, device)
                        if not _rank0():
                            continue
                        token_ids = _bits_to_token_ids(gathered, bits_per_token)
                        gen_wavs = _decode_speech_wavs(
                            speech_tok, token_ids, text_seq_len, speaker_seq_len, speech_offset, speech_vocab_sz,
                        )
                        task_dir = run_dir / spec.tag / 'salmon' / salmon_task
                        task_dir.mkdir(parents=True, exist_ok=True)
                        samples = []
                        for i, w in enumerate(gen_wavs):
                            wav_name = f'gen_{i:04d}.wav'
                            _write_wav(task_dir / wav_name, w, sample_rate)
                            samples.append({'gen_wav': wav_name})
                        _write_samples(task_dir, samples)
                    continue

                # ---------------------------------------------------------------
                # stt: per-partition, text-only (no generated audio)
                # ---------------------------------------------------------------
                if task == 'stt':
                    for p in stt_partitions:
                        loader = loaders['stt'][p]
                        local_gen: List[torch.Tensor] = []
                        local_ref: List[torch.Tensor] = []
                        for batch in loader:
                            x_full = _prepare_x_full(batch, task, sequence_len, bits_per_token, text_seq_len, device)
                            B = x_full.size(0)
                            with autocast(enabled=use_amp, dtype=amp_dtype):
                                bits = _generate_batch(
                                    raw_model, proc, cfg, x_full, task, B,
                                    sequence_len, bits_per_token, spec,
                                    terminal_sigma, entropic_blend_alpha, entropy_run_dir,
                                    sampler_cache,
                                )
                            local_gen.append(bits)
                            local_ref.append((x_full > 0.5).to(torch.long))

                        gen_gathered = _gather_bits(local_gen, sequence_len, device)
                        ref_gathered = _gather_bits(local_ref, sequence_len, device)
                        if not _rank0():
                            continue

                        gen_texts = _decode_texts(text_tok, _bits_to_token_ids(gen_gathered, bits_per_token), text_seq_len, speech_offset, speech_vocab_sz)
                        ref_texts = _decode_texts(text_tok, _bits_to_token_ids(ref_gathered, bits_per_token), text_seq_len, speech_offset, speech_vocab_sz)

                        task_dir = run_dir / spec.tag / f'stt_{p}'
                        ref_dir = run_dir / f'stt_{p}'
                        ref_cache = stt_ref_cache_by_partition.get(p)
                        samples = []
                        for i in range(len(gen_texts)):
                            sample = {'gen_text': gen_texts[i], 'ref_text': ref_texts[i]}
                            if write_ref and ref_cache and i < len(ref_cache):
                                ref_dir.mkdir(parents=True, exist_ok=True)
                                ref_wav_name = f'ref_{i:04d}.wav'
                                _write_wav(ref_dir / ref_wav_name, ref_cache[i], sample_rate)
                                sample['ref_wav'] = f'stt_{p}/{ref_wav_name}'
                            samples.append(sample)
                        _write_samples(task_dir, samples)
                    continue

                # ---------------------------------------------------------------
                # joint / tts / cont
                # ---------------------------------------------------------------
                ref_gathered = None
                if task == 'joint':
                    my_idx = _shard_indices(num_uncond_samples, world_size, rank)
                    remaining = len(my_idx)
                    local_gen = []
                    while remaining > 0:
                        B = min(gen_batch_size, remaining)
                        remaining -= B
                        x_full = torch.zeros(B, sequence_len, device=device, dtype=torch.float32)
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            bits = _generate_batch(
                                raw_model, proc, cfg, x_full, task, B,
                                sequence_len, bits_per_token, spec,
                                terminal_sigma, entropic_blend_alpha, entropy_run_dir,
                                sampler_cache,
                            )
                        local_gen.append(bits)
                    gen_gathered = _gather_bits(local_gen, sequence_len, device)
                else:
                    loader = loaders[task]
                    local_gen = []
                    local_ref = [] if task == 'tts' else None
                    for batch in loader:
                        x_full = _prepare_x_full(batch, task, sequence_len, bits_per_token, text_seq_len, device)
                        B = x_full.size(0)
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            bits = _generate_batch(
                                raw_model, proc, cfg, x_full, task, B,
                                sequence_len, bits_per_token, spec,
                                terminal_sigma, entropic_blend_alpha, entropy_run_dir,
                                sampler_cache,
                            )
                        local_gen.append(bits)
                        if task == 'tts':
                            local_ref.append((x_full > 0.5).to(torch.long))
                    gen_gathered = _gather_bits(local_gen, sequence_len, device)
                    if task == 'tts':
                        ref_gathered = _gather_bits(local_ref, sequence_len, device)

                if not _rank0():
                    continue

                gen_token_ids = _bits_to_token_ids(gen_gathered, bits_per_token)
                gen_texts = (
                    _decode_texts(text_tok, gen_token_ids, text_seq_len, speech_offset, speech_vocab_sz)
                    if task in ('joint', 'cont') else None
                )
                gen_wavs = _decode_speech_wavs(
                    speech_tok, gen_token_ids, text_seq_len, speaker_seq_len, speech_offset, speech_vocab_sz,
                )
                ref_texts = None
                if task == 'tts':
                    ref_token_ids = _bits_to_token_ids(ref_gathered, bits_per_token)
                    ref_texts = _decode_texts(text_tok, ref_token_ids, text_seq_len, speech_offset, speech_vocab_sz)

                task_dir = run_dir / spec.tag / task
                task_dir.mkdir(parents=True, exist_ok=True)
                ref_dir = run_dir / task
                samples = []
                for i in range(gen_token_ids.size(0)):
                    sample: Dict[str, Any] = {}
                    if gen_texts is not None:
                        sample['gen_text'] = gen_texts[i]
                    if ref_texts is not None:
                        sample['ref_text'] = ref_texts[i]

                    wav_name = f'gen_{i:04d}.wav'
                    _write_wav(task_dir / wav_name, gen_wavs[i], sample_rate)
                    sample['gen_wav'] = wav_name

                    if write_ref and task == 'tts' and tts_ref_cache and i < len(tts_ref_cache):
                        ref_wav_name = f'ref_{i:04d}.wav'
                        _write_wav(ref_dir / ref_wav_name, tts_ref_cache[i], sample_rate)
                        sample['ref_wav'] = f'{task}/{ref_wav_name}'

                    samples.append(sample)
                _write_samples(task_dir, samples)

    barrier()
    if _rank0():
        _dbg(f"Done. Samples written to: {run_dir}")
        _dbg(f"To evaluate the generated samples, run evaluation/evaluation_drivers/textaudio_eval.py")
    return run_dir


def evaluate_textaudio_generate(args, cfg, model, device, rank0: bool, ddp_active: bool, dist_info) -> Optional[Path]:
    """Entry point matching the run_eval.py driver-call convention."""
    rank = int(dist_info.rank) if ddp_active else 0
    world_size = int(dist_info.world_size) if ddp_active else 1
    return generate_textaudio(cfg, model, device, rank=rank, world_size=world_size)
