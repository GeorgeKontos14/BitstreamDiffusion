from __future__ import annotations

import json
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast

from jiwer import wer

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.distributed as dist
except Exception:
    dist = None

from utils.textaudio_utils import _fixed_mask, _safe_decode, UTMOS
from utils.textaudio_report import build_textaudio_report
from utils.model_utils import unwrap_model

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
        print(f"[TextAudio][rank{_rank()}/{_world()}] {msg}", flush=True)
    elif _rank0():
        print(f"[TextAudio][rank{_rank()}/{_world()}] {msg}", flush=True)

def _tb(trainer) -> Optional[Any]:
    tb = getattr(trainer, "tb", None)
    if tb is not None:
        return tb
    tb = getattr(trainer, "tb_manager", None)
    if tb is not None:
        return tb
    return None

def _global_step(trainer, epoch: int) -> int:
    gs = getattr(trainer, "global_step", None)
    if gs is None:
        return int(epoch)
    try:
        return int(gs)
    except Exception:
        return int(epoch)

# -----------------------------------------------------------------------------
# Writing to disk
# -----------------------------------------------------------------------------

def _write_wav(path, arr, sr: int) -> None:
    arr = np.asarray(arr, dtype=np.float32).flatten()
    pcm = (arr*32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    
# -----------------------------------------------------------------------------
# Config resolution
# -----------------------------------------------------------------------------
@dataclass
class _ResolvedTextAudio:
    enabled: bool
    every_k_epochs: int
    run_on_sanity: bool
    split: str

    num_samples: int
    sampler: str
    terminal_sigma: float
    num_steps: int
    entropic_blend_alpha: float
    entropy_run_dir: Optional[str]
    seed: int

    whisper_model: str
    sample_rate: int

    bits_per_token: int
    text_seq_len: int
    speaker_seq_len: int
    speech_seq_len: int
    speech_offset: int
    speech_vocab_size: int
    sequence_len: int

    use_amp: bool

def _first_scalar(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) > 0 else default
    return x

def _sanitize_sampler_name(x, default: str = "ddim_entropic") -> str:
    if x is None:
        return default

    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) > 0 else default

    s = str(x).strip()

    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if "," in inner:
            inner = inner.split(",", 1)[0].strip()
        inner = inner.strip().strip("'").strip('"')
        if inner:
            s = inner

    return s or default

def _resolve_cfg(cfg: Any) -> _ResolvedTextAudio:
    train = getattr(cfg, 'train', None)
    data = getattr(cfg, 'data', None)

    c = getattr(train, 'textaudio', None)
    gen = getattr(train, 'generation', None)

    enabled = bool(getattr(c, 'enabled', False))

    run_on_sanity = False
    if c is not None and getattr(c, 'run_on_sanity', None) is not None:
        run_on_sanity = bool(getattr(c, 'run_on_sanity', None))

    k = None
    if c is not None:
        k = getattr(c, "every_k_epochs", None)
        if k is None:
            k = getattr(c, "every_epochs", None)
        if k is None:
            k = getattr(c, "every", None)
    if k is None:
        k = 10 if enabled else 1
    every_k_epochs = max(1, int(k))   

    split = 'val'
    if c is not None:
        s = getattr(c, 'split', None)
        if s is not None:
            split = str(s).lower()

    def pick(name:str, gen_name: str, default): 
        if c is not None and getattr(c, name, None) is not None:
            return getattr(c, name)
        if gen is not None and getattr(gen, gen_name, None) is not None:
            return getattr(gen, gen_name)
        return default
    
    num_samples = int(pick('num_samples', 'num_samples', 64))
    raw_sampler = None
    if c is not None:
        raw_sampler = getattr(c, "sampler", None)
        if raw_sampler is None:
            raw_sampler = getattr(c, "samplers", None)
    if raw_sampler is None and gen is not None:
        raw_sampler = getattr(gen, "sampler", None)
        if raw_sampler is None:
            raw_sampler = getattr(gen, "samplers", None)
    sampler = _sanitize_sampler_name(raw_sampler, default="ddim_entropic")

    terminal_sigma = getattr(c, "terminal_sigma", None) if c is not None else None
    if terminal_sigma is None and c is not None:
        terminal_sigma = getattr(c, "terminal_sigmas", None)
    if terminal_sigma is None and gen is not None:
        terminal_sigma = getattr(gen, "terminal_sigmas", None)
    terminal_sigma = float(_first_scalar(terminal_sigma, 0.08))

    num_steps = int(
        (getattr(c, "num_steps", None) if c is not None else None)
        or (getattr(c, "num_sampling_steps", None) if c is not None else None)
        or (getattr(gen, "num_sampling_steps", None) if gen is not None else None)
        or 64
    )

    entropic_blend_alpha = float(pick("entropic_blend_alpha", "entropic_blend_alpha", 0.0))

    entropy_run_dir = None
    if c is not None:
        entropy_run_dir = getattr(c, "entropy_run_dir", None)
        if entropy_run_dir is None:
            entropy_run_dir = getattr(c, "entropy_ckpt_path", None)
    if entropy_run_dir is None and gen is not None:
        entropy_run_dir = getattr(gen, "entropy_ckpt_path", None)

    seed = int(pick("seed", "seed", 42))

    whisper_model = "openai/whisper-medium"
    if c is not None and getattr(c, "whisper_model", None) is not None:
        whisper_model = str(getattr(c, "whisper_model"))
 
    sample_rate = int(getattr(data, "sample_rate", 16000)) if data is not None else 16000    

    bits_per_token = int(getattr(data, "bits_per_token", 18)) if data is not None else 18
    text_seq_len = int(getattr(data, "text_seq_len", 168)) if data is not None else 168
    speaker_seq_len = int(getattr(data, "speaker_seq_len", 32)) if data is not None else 32
    speech_seq_len = int(getattr(data, "speech_seq_len", 800)) if data is not None else 800
    speech_offset = int(getattr(data, "speech_offset", 204115)) if data is not None else 204115
    speech_vocab_size = int(getattr(data, "speech_vocab_size", 46656)) if data is not None else 46656
    sequence_len = int(getattr(data, "sequence_len", 1000)) if data is not None else 1000    

    use_amp = bool(getattr(c, "use_amp", True)) if c is not None else True

    return _ResolvedTextAudio(
        enabled=enabled,
        every_k_epochs=every_k_epochs,
        run_on_sanity=run_on_sanity,
        split=split,
        num_samples=num_samples,
        sampler=sampler,
        terminal_sigma=terminal_sigma,
        num_steps=num_steps,
        entropic_blend_alpha=entropic_blend_alpha,
        entropy_run_dir=entropy_run_dir,
        seed=seed,
        whisper_model=whisper_model,
        sample_rate=sample_rate,
        bits_per_token=bits_per_token,
        text_seq_len=text_seq_len,
        speaker_seq_len=speaker_seq_len,
        speech_seq_len=speech_seq_len,
        speech_offset=speech_offset,
        speech_vocab_size=speech_vocab_size,
        sequence_len=sequence_len,
        use_amp=use_amp,
    )

TASKS = ['joint', 'tts', 'stt', 'cont']

class TextAudioEvaluator:
    def __init__(self, whisper_model: str = 'openai/whisper-medium', sr=16_000):
        self._whisper_name = whisper_model
        self._sr = sr
        self._asr = None
        self._utmos = None

    def _ensure_asr(self, device):
        if self._asr is None:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            _dbg(f'Loading {self._whisper_name} on {device}')
            _model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._whisper_name, torch_dtype=torch.float16
            ).to(device)
            _proc = AutoProcessor.from_pretrained(self._whisper_name)
            self._asr = pipeline(
                'automatic-speech-recognition',
                model=_model,
                tokenizer = _proc.tokenizer,
                feature_extractor=_proc.feature_extractor,
                torch_dtype=torch.float16,
                device=device
            )
        return self._asr


    def transcribe(self, wavs, device) -> list[str]:
        asr = self._ensure_asr(device)
        inputs, valid_idx = [], []
        for i, wav in enumerate(wavs):
            if wav is None:
                continue
            arr = np.asarray(wav, dtype=np.float32).flatten()
            if arr.ndim > 1:
                arr = arr[0]
            inputs.append({'array': arr, 'sampling_rate': self._sr})
            valid_idx.append(i)
        
        results = [""]*len(wavs)
        if inputs:
            try:
                preds = asr(inputs, batch_size=min(64, len(inputs)))
                for idx, pred in zip(valid_idx, preds):
                    results[idx] = pred['text'].strip()
            except Exception as e:
                _dbg(f'ASR batch failed: {e}')

        return results
    
    def _ensure_utmos(self):
        if self._utmos is None:
            _dbg('Loading UTMOS')
            self._utmos = UTMOS(sr=self._sr, use_gpu=torch.cuda.is_available())

        return self._utmos
    
    def utmos_score(self, wavs) -> float:
        model = self._ensure_utmos()
        
        scores = []
        for wav in wavs:
            if wav is None:
                continue
            arr = np.asarray(wav, dtype=np.float32).flatten()
            if arr.ndim > 1:
                arr = arr[0]
            try:
                scores.append(float(model.score(arr)))
            except Exception as e:
                _dbg(f'UTMOS score failed: {e}')

        return float(np.mean(scores)) if scores else float('nan')
    
    @staticmethod
    def word_error_rate(refs, hyps) -> float:
        refs = [r.lower() for r in refs]
        hyps = [h.lower() for h in hyps]
        if not refs:
            return float('nan')
        return float(wer(refs, hyps))
    
    def evaluate_task(
        self, task:str, gen_texts:List[str], ref_texts: List[str], gen_wavs, device
    ) -> Dict[str, float]:
        metrics = {}
        transcriptions = None

        if task == 'joint':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['wer'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['utmos'] = self.utmos_score(gen_wavs)
        elif task == 'tts':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['wer'] = self.word_error_rate(ref_texts, transcriptions)
            metrics['utmos'] = self.utmos_score(gen_wavs)
        elif task == 'stt':
            metrics['wer'] = self.word_error_rate(ref_texts, gen_texts)
        elif task == 'cont':
            metrics['utmos'] = self.utmos_score(gen_wavs)

        return metrics, transcriptions
    
# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------
class TextAudioCallback:
    run_on_all_ranks=True
    
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._ddim_sampler = None
        self._evaluator = None
        self._last_run_key = None

    def _ensure_evaluator(self, r: _ResolvedTextAudio) -> None:
        if self._evaluator is not None:
            return
        self._evaluator = TextAudioEvaluator(r.whisper_model, r.sample_rate)

    def _should_run(self, epoch: int, r: _ResolvedTextAudio) -> bool:
        if not r.enabled:
            return False
        # epoch < 0 → sanity check run
        if int(epoch) < 0:
            return bool(r.run_on_sanity)
        k = max(1, int(r.every_k_epochs))
        return ((int(epoch) + 1) % k) == 0
    
    @torch.no_grad()
    def _sample_real_data(self, trainer, B: int, split: str) -> Optional[torch.Tensor]:
        loader = getattr(trainer, f'{split}_loader', None)
        if loader is None:
            return None
        dataset = loader.dataset
        if len(dataset) == 0:
            return None

        n = min(B, len(dataset))
        chunks = []
        for i in range(n):
            sample = dataset[i]
            if isinstance(sample, (tuple, list)):
                sample = sample[0]
            sample = sample.view(-1).to(device=trainer.device, dtype=torch.float32, non_blocking=True)
            chunks.append(sample.unsqueeze(0))

        return torch.cat(chunks, dim=0).contiguous()
    
    def _get_sampler(self, trainer, r: _ResolvedTextAudio):
        raw = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        if "ddim" in r.sampler:
            if self._ddim_sampler is None:
                from diffusion.continuous.samplers import DDIMSampler
                self._ddim_sampler = DDIMSampler(
                    raw, trainer.proc, self.cfg,
                )
            return self._ddim_sampler
        return trainer.sampler

    def _generate_task(
        self, trainer, x_full: torch.Tensor, task_id: int, B: int, r: _ResolvedTextAudio
    ) -> torch.Tensor:
        cond_mask = _fixed_mask(
            self.cfg, B, r.sequence_len, task_id, device=trainer.device, bits_per_token=r.bits_per_token
        )
        schedule = 'entropic' if 'entropic' in r.sampler else 'karras'
        sampler_obj = self._get_sampler(trainer, r)
        
        entropy_run_dir = r.entropy_run_dir
        if entropy_run_dir is None:
            entropy_run_dir = str(getattr(trainer, 'run_dir', '.'))

        cond_kwargs = dict(
            conditioning_prefix_full=x_full,
            cond_prefix_mask=cond_mask,
            guidance_scale=0.0
        )

        _, probs = sampler_obj.sample(
            B, r.sequence_len,
            schedule=schedule,
            num_steps=r.num_steps,
            entropic_blend_alpha=r.entropic_blend_alpha,
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=r.terminal_sigma,
            return_probs=True,
            progress=True,
            **cond_kwargs
        )
        
        bits = (probs > 0.5).to(torch.long)
        bits[cond_mask] = (x_full[cond_mask] > 0.5).to(torch.long)
        return bits

    def _bits_to_token_ids(
        self, bits: torch.Tensor, r: _ResolvedTextAudio
    ) -> torch.Tensor:
        B = bits.size(0)
        tok_bits = bits.view(B, -1, r.bits_per_token)
        shifts = torch.arange(
            r.bits_per_token-1,-1,-1, device=bits.device
        )
        return (tok_bits * (2**shifts)).sum(dim=-1).long()
    
    def _decode_texts(
        self, trainer, token_ids: torch.Tensor, r: _ResolvedTextAudio
    ) -> List[str]:
        PAD_TEXT = r.speech_offset+r.speech_vocab_size
        ids = token_ids[:, :r.text_seq_len]
        pad = ids == PAD_TEXT
        ids_clean = ids.clone()
        ids_clean[pad] = 0

        texts = []
        for i in range(ids.size(0)):
            valid = ids_clean[i][~pad[i]].cpu().tolist()
            texts.append(_safe_decode(trainer.text_tok, valid))
        return texts
    
    def _decode_speech_wavs(
        self, trainer, token_ids: torch.Tensor, r: _ResolvedTextAudio
    ):
        PAD_SPEECH = r.speech_offset+r.speech_vocab_size+1
        global_ids = token_ids[:, r.text_seq_len+r.speaker_seq_len:]
        pad = global_ids == PAD_SPEECH
        local_ids = (global_ids-r.speech_offset).clamp(0, r.speech_vocab_size-1)

        wavs = []
        for i in range(local_ids.size(0)):
            valid = local_ids[i][~pad[i]]
            wav = trainer.speech_tok.decode([valid.unsqueeze(0).unsqueeze(-1)], posthoc_bottleneck=True)
            wav = wav.squeeze(0).cpu().float().squeeze(0).numpy()
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav = wav / peak
            wavs.append(wav)
        return wavs
    
    def _log_scalars(
        self, trainer, metrics: Dict[str, float], tag: str, step: int
    ):
        for name, value in metrics.items():
            if math.isnan(value):
                continue
            key = f"textaudio/{tag}/{name}"
            w = getattr(trainer, 'writer', None)
            if w is not None:
                try:
                    w.add_scalar(key, value, step)
                except TypeError:
                    w.add_scalar(key, value)
            
            if getattr(trainer, 'use_wandb', False):
                try: 
                    trainer._log_wandb({key:value})
                except Exception:
                    pass

            _dbg(f'{key} = {value:.4f}')

    def _log_samples(
        self, trainer, task, tag: str, step: int,
        gen_texts, ref_texts, gen_wavs, ref_wavs, transcriptions,
        r: _ResolvedTextAudio
    ):
        has_gen_text = task in ('joint', 'stt', 'cont')
        has_ref_text = task in ('tts', 'stt')
        has_gen_audio = task in ('joint', 'tts', 'cont')
        has_ref_audio = task in ('tts', 'stt')
        has_whisper = transcriptions is not None

        w = getattr(trainer, 'writer', None)
        n = r.num_samples
        prefix = f'textaudio/{tag}'


        if w is not None:
            lines = []
            for i in range(n):
                lines.append(f'Sample {i}')
                if has_gen_text:
                    lines.append(f' gen_text:   {repr(gen_texts[i])}')
                if has_ref_text:
                    lines.append(f' ref_text:   {repr(ref_texts[i])}')
                if has_whisper:
                    lines.append(f' whisper:    {repr(transcriptions[i])}')
                lines.append('')
            body = '\n'.join(lines)
            try:
                w.add_text(f'{prefix}/samples', f'```\n{body}\n```', step)
            except Exception:
                pass

            if has_gen_audio:
                for i, wav in enumerate(gen_wavs):
                    if wav is None:
                        continue
                    w.add_audio(
                        f'{prefix}/gen_audio_{i}', wav, step, sample_rate = r.sample_rate 
                    )
            if has_ref_audio:
                for i, wav in enumerate(ref_wavs):
                    if wav is None:
                        continue
                    w.add_audio(
                        f'{prefix}/ref_audio_{i}', wav, step, sample_rate = r.sample_rate 
                    )                
            
        if getattr(trainer, 'use_wandb', False) and wandb is not None:
            try:
                columns = ['idx']
                if has_ref_text:
                    columns.append('ref_text')
                if has_gen_text:
                    columns.append('gen_text')
                if has_ref_audio:
                    columns.append('ref_audio')
                if has_gen_audio:
                    columns.append('gen_audio')
                if has_whisper:
                    columns.append('whisper')

                table = wandb.Table(columns=columns)

                for i in range(n):
                    row = [i]
                    if has_ref_text:
                        row.append(ref_texts[i])
                    if has_gen_text:
                        row.append(gen_texts[i])
                    if has_ref_audio:
                        row.append(wandb.Audio(ref_wavs[i], sample_rate=r.sample_rate))
                    if has_gen_audio:
                        row.append(wandb.Audio(gen_wavs[i], sample_rate=r.sample_rate))
                    if has_whisper:
                        row.append(transcriptions[i])
                        
                    table.add_data(*row)
                
                trainer._log_wandb({f'textaudio/{tag}/samples': table})
            except Exception as e:
                _dbg(f'W&B table log failed {e}')

    def _save_to_disk(
        self, trainer, all_results: Dict[str, dict], r: _ResolvedTextAudio, step: int, epoch: int
    ) -> None:
        run_dir = Path(str(getattr(trainer, 'run_dir', 'runs/default')))
        base_dir = run_dir / 'textaudio'
        save_dir = base_dir / f'step_{step:09d}'
        save_dir.mkdir(parents=True, exist_ok=True)

        data_cfg = getattr(self.cfg, 'data', None)

        report_data = {
            'header': {
                'title': 'Single-stream Joint Audio+Text Generation',
                'experiment': str(getattr(self.cfg, 'experiment', 'unknown')) if data_cfg else 'unknown',
                'text_tokenizer': str(
                    getattr(data_cfg, 'text_tokenizer', 'o200k_base')
                ) if data_cfg else 'o200k_base',
                'speaker_tokenizer': str(
                    getattr(data_cfg, 'speaker_tokenizer', 'bicodec')
                ) if data_cfg else 'bicodec',
                'speech_tokenizer': str(
                    getattr(data_cfg, 'speech_tokenizer', 'stabilityai/stable-codec-speech-16k')
                ) if data_cfg else 'stabilityai/stable-codec-speech-16k',
                'speech_tokenizer_bottleneck': str(
                    getattr(data_cfg,'speech_tokenizer_bottleneck', '1x46656_400bps')
                ) if data_cfg else '1x46656_400bps',
                'bits_per_token': r.bits_per_token,
                'sequence_layout': {
                    'text_tokens': r.text_seq_len,
                    'speaker_tokens': r.speaker_seq_len,
                    'speech_tokens': r.speech_seq_len,
                    'total_tokens': r.sequence_len
                },
                'split': r.split,
                'step': step,
                'epoch': epoch,
                'num_samples': r.num_samples,
                'sampler': {
                    'name': r.sampler,
                    'num_steps': r.num_steps,
                    'terminal_sigma': r.terminal_sigma,
                },
                'sample_rate': r.sample_rate,
            },
            'tasks': {},
        }

        for task, res in all_results.items():
            task_dir = save_dir / task
            task_dir.mkdir(exist_ok=True)

            gen_texts = res.get('gen_texts', [])
            ref_texts = res.get('ref_texts', [])
            gen_wavs = res.get('gen_wavs', [])
            ref_wavs = res.get('ref_wavs')
            transcriptions = res.get('transcriptions')
            metrics = res.get('metrics', {})

            samples = []
            n = r.num_samples

            for i in range(n):
                sample = {'idx': i}

                if task in ('joint', 'stt', 'cont'):
                    sample['gen_text'] = gen_texts[i]
                if task in ('tts', 'stt'):
                    sample['ref_text'] = ref_texts[i]

                if task in ('joint', 'tts', 'cont'):
                    wav_rel = f'{task}/gen_{i:04d}.wav'
                    _write_wav(save_dir / wav_rel, gen_wavs[i], r.sample_rate)
                    sample['gen_wav'] = wav_rel
                
                if task in ('tts', 'stt'):
                    wav_rel = f'{task}/ref_{i:04d}.wav'
                    _write_wav(save_dir / wav_rel, ref_wavs[i], r.sample_rate)
                    sample['ref_wav'] = wav_rel

                if task in ('joint', 'tts'):
                    sample['whisper'] = transcriptions[i]

                samples.append(sample)

            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, float) and math.isnan(v):
                    clean_metrics[k] = None
                else:
                    clean_metrics[k] = v

            report_data['tasks'][task] = {
                'metrics': clean_metrics,
                'samples': samples
            }

        with open(save_dir / 'data.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        
        html_path = build_textaudio_report(save_dir)
 
        _dbg(f"Saved to {save_dir}")
        _dbg(f"See report: {html_path}")

    @torch.compiler.disable
    @torch.no_grad()
    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        r = _resolve_cfg(self.cfg)
        
        if not self._should_run(epoch, r):
            return

        run_key = (int(epoch), _global_step(trainer, epoch))
        
        if self._last_run_key == run_key:
            return
        self._last_run_key = run_key

        if _rank0():
            t0 = time.perf_counter()

        B = r.num_samples
        raw = unwrap_model(trainer.model)

        # ── ALL ranks: EMA, eval, generate ──────────────────
        try:
            trainer.model.eval()
            trainer.ema.apply(trainer.model)   # match _validate_epoch

            torch.manual_seed(r.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(r.seed)

            x_full = self._sample_real_data(trainer, B, r.split)
            has_data = x_full is not None

            if has_data:
                B = x_full.size(0)
                if _rank0():
                    _dbg(
                        f"START epoch={epoch} split={r.split} "
                        f"B={B} steps={r.num_steps} sampler={r.sampler} "
                        f"sigma={r.terminal_sigma}"
                    )

                amp_dtype = getattr(trainer, "amp_dtype", torch.float16)
                task_bits = {}

                with autocast(enabled=r.use_amp, dtype=amp_dtype):
                    for task_id, task in enumerate(TASKS):
                        if _rank0():
                            _dbg(f"  {r.split}_{task}: generating {B} samples")
                        try:
                            bits = self._generate_task(trainer, x_full, task_id, B, r)
                            task_bits[task] = bits
                        except Exception as e:
                            if _rank0():
                                _dbg(f"  {r.split}_{task}: generation failed: {e}")
            else:
                task_bits = {}
                if _rank0():
                    _dbg(f"No data from {r.split} split")

        finally:
            trainer.ema.restore(trainer.model)   # match _validate_epoch
            trainer.model.train()

        # ── RANK 0 only: decode, evaluate, save ─────────────
        if not _rank0():
            return

        if has_data and task_bits:
            step = _global_step(trainer, epoch)
            self._ensure_evaluator(r)
            all_results = {}

            for task, bits in task_bits.items():
                _dbg(f"  {r.split}_{task}: decoding+evaluating")
                try:
                    needs_gen_text = task in ("joint", "stt", "cont")
                    needs_gen_wav  = task in ("joint", "tts", "cont")
                    needs_ref      = task in ("tts", "stt")

                    token_ids  = self._bits_to_token_ids(bits, r)
                    gen_texts  = self._decode_texts(trainer, token_ids, r) if needs_gen_text else []
                    gen_wavs   = self._decode_speech_wavs(trainer, token_ids, r) if needs_gen_wav else []

                    ref_texts = []
                    ref_wavs  = None
                    if needs_ref:
                        ref_token_ids = self._bits_to_token_ids(
                            (x_full > 0.5).to(torch.long), r,
                        )
                        ref_texts = self._decode_texts(trainer, ref_token_ids, r)
                        ref_wavs  = self._decode_speech_wavs(trainer, ref_token_ids, r)

                    metrics, transcriptions = self._evaluator.evaluate_task(
                        task, gen_texts, ref_texts, gen_wavs, trainer.device,
                    )
                except Exception as e:
                    _dbg(f"  {r.split}_{task}: eval failed: {e}")
                    continue

                all_results[task] = {
                    "metrics": metrics, "gen_texts": gen_texts,
                    "ref_texts": ref_texts, "gen_wavs": gen_wavs,
                    "ref_wavs": ref_wavs, "transcriptions": transcriptions,
                }

                tag = f"{r.split}_{task}"
                self._log_scalars(trainer, metrics, tag, step)
                self._log_samples(
                    trainer, task, tag, step,
                    gen_texts, ref_texts, gen_wavs, ref_wavs, transcriptions, r,
                )

            if all_results:
                try:
                    self._save_to_disk(trainer, all_results, r, step, epoch)
                except Exception as e:
                    _dbg(f"Local save failed: {e}")

        elapsed = time.perf_counter() - t0
        w = getattr(trainer, "writer", None)
        if w is not None:
            try:
                w.add_scalar("textaudio/timing_sec", elapsed, _global_step(trainer, epoch))
            except Exception:
                pass
        _dbg(f"END ({elapsed:.1f}s)")