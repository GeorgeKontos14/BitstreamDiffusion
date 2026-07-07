import re

import numpy as np

import torch
from ml_collections import config_dict

from jiwer import wer, cer

from typing import Any, Dict, List

# Task IDs
UNCONDITIONAL = 0
TEXT_TO_SPEECH = 1
SPEECH_TO_TEXT = 2
SPEECH_CONTINUATION = 3
CONDITIONAL_TASKS = [TEXT_TO_SPEECH, SPEECH_TO_TEXT, SPEECH_CONTINUATION]

def _sample_tasks_and_cond_masks(
    cfg: config_dict.ConfigDict, B: int, S: int, device: torch.device, bits_per_token: int = 18
) -> tuple[torch.Tensor, torch.Tensor]:
    text_len = int(getattr(cfg.data, 'text_seq_len', 168))
    speaker_len = int(getattr(cfg.data, 'speaker_seq_len', 32))

    # Number of prefix tokens for speech continuation
    prefix_len = int(getattr(cfg.cond, 'continuation_prefix', 75))

    text_end = text_len
    speaker_end = text_len+speaker_len
    cont_end = speaker_end+prefix_len
    
    task_weights = torch.tensor([
        float(getattr(cfg.cond, 'unconditional_rate', 0.25)),
        float(getattr(cfg.cond, 'texttospeech_rate', 0.25)),
        float(getattr(cfg.cond, 'speechtotext_rate', 0.25)),
        float(getattr(cfg.cond, 'continuation_rate', 0.25))
    ])
    assert torch.sum(task_weights) == 1, 'Masking ratios must sum up to 1'
    task_ids = torch.multinomial(task_weights, B, replacement=True)

    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    for task in CONDITIONAL_TASKS:
        sel = task_ids == task
        if not sel.any():
            continue
        
        if task == TEXT_TO_SPEECH:
            mask[sel, :speaker_end*bits_per_token] = True
        elif task == SPEECH_TO_TEXT:
            mask[sel, text_end*bits_per_token:] = True
        else: # Continuation
            mask[sel, text_end*bits_per_token:cont_end*bits_per_token] = True
    
    return task_ids, mask

def _fixed_mask(
    cfg: config_dict.ConfigDict, B: int, S: int, task: int, device: torch.device, bits_per_token: int = 18
) -> torch.Tensor:
    text_len = int(getattr(cfg.data, 'text_seq_len', 168))
    speaker_len = int(getattr(cfg.data, 'speaker_seq_len', 32))

    # Number of prefix tokens for speech continuation
    prefix_len = int(getattr(cfg.cond, 'continuation_prefix', 75))

    text_end = text_len
    speaker_end = text_len+speaker_len
    cont_end = speaker_end+prefix_len

    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    if task == TEXT_TO_SPEECH:
        mask[:, :speaker_end*bits_per_token] = True
    elif task == SPEECH_TO_TEXT:
        mask[:, text_end*bits_per_token:] = True
    elif task == SPEECH_CONTINUATION:
        mask[:, text_end*bits_per_token:cont_end*bits_per_token] = True
    
    return mask
    
def _safe_decode(enc, token_ids):
    parts = []
    for t in token_ids:
        try:
            parts.append(enc.decode_single_token_bytes(t))
        except KeyError:
            parts.append(b'<unk>')
    return b"".join(parts).decode("utf-8", errors="replace")

# Code taken from https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
# Pasted to bypass import issues
class UTMOS:

    def __init__(self, sr=16000, use_gpu=True):
        """
        Args:
            sr (int): Sampling rate.
            use_gpu (bool): Whether to use GPU.
        """
        self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.predictor.eval()
        self.predictor.to(self.device)
        self.sr = sr
    
    def score(self, gen_wav):
        """
        Args:
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: UTMOS score.
        """
        gen_wav = torch.from_numpy(gen_wav).unsqueeze(0).to(self.device).float()
        score = self.predictor(gen_wav, self.sr)
        return score[0].item()
    
class TextAudioEvaluator:
    def __init__(self, whisper_model: str = 'openai/whisper-medium', sr=16_000, _dbg_func=None):
        self._whisper_name = whisper_model
        self._sr = sr
        self._asr = None
        self._utmos = None
        self._dbg_func = _dbg_func

    def _dbg(self, msg: str):
        if self._dbg_func:
            self._dbg_func(msg)
        else:
            print(msg)

    def _ensure_asr(self, device):
        if self._asr is None:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            self._dbg(f'Loading {self._whisper_name} on {device}')
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
                device=device,
                chunk_length_s=30,
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
                self._dbg(f'ASR batch failed: {e}')

        return results
    
    def _ensure_utmos(self):
        if self._utmos is None:
            self._dbg('Loading UTMOS')
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
                self._dbg(f'UTMOS score failed: {e}')

        return float(np.mean(scores)) if scores else float('nan')
    
    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def word_error_rate(self, refs, hyps) -> float:
        refs = [self.normalize_text(r) for r in refs]
        hyps = [self.normalize_text(h) for h in hyps]
        if not refs:
            return float('nan')
        return float(wer(refs, hyps))

    def character_error_rate(self, refs, hyps) -> float:
        refs = [self.normalize_text(r) for r in refs]
        hyps = [self.normalize_text(h) for h in hyps]
        if not refs:
            return float('nan')
        return float(cer(refs, hyps))
    
    def evaluate_task_brief(
        self, task:str, gen_texts:List[str], ref_texts: List[str], gen_wavs, device
    ) -> Dict[str, float]:
        metrics = {}
        transcriptions = None

        if task == 'joint':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['Cross-Modal WER'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['Cross-Modal CER'] = self.character_error_rate(gen_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
        elif task == 'tts':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['ASR-WER'] = self.word_error_rate(ref_texts, transcriptions)
            metrics['ASR-CER'] = self.character_error_rate(ref_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
        elif task == 'stt':
            metrics['WER'] = self.word_error_rate(ref_texts, gen_texts)
            metrics['CER'] = self.character_error_rate(ref_texts, gen_texts)
        elif task == 'cont':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
            metrics['WER'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['CER'] = self.character_error_rate(gen_texts, transcriptions)

        return metrics, transcriptions