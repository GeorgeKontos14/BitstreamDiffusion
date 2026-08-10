import math
import re
from pathlib import Path

import numpy as np

import torch
from ml_collections import config_dict

from jiwer import wer, cer

from scipy.linalg import sqrtm

from typing import Any, Dict, List

from transformers import AutoFeatureExtractor, AutoTokenizer, WavLMModel, AutoModelForCausalLM

from utils.speaker_verification import ECAPA_TDNN_SMALL

# Task IDs
UNCONDITIONAL = 0
TEXT_TO_SPEECH = 1
SPEECH_TO_TEXT = 2
SPEECH_CONTINUATION = 3
CONDITIONAL_TASKS = [TEXT_TO_SPEECH, SPEECH_TO_TEXT, SPEECH_CONTINUATION]

# Continuation samples are prefixed with this many seconds of reference
# (non-generated) audio -- matches scripts/multimodal_textaudio/prepare_test.py's
# CONT_TRIM_SECONDS used when building the continuation cache.
CONT_TRIM_SECONDS = 3.0


def _trim_prefix_seconds(wav, seconds: float, sr: int):
    """Drops the first `seconds` of a waveform (the reference prefix for
    continuation samples). Returns None if nothing is left afterward."""
    if wav is None:
        return None
    arr = np.asarray(wav, dtype=np.float32).flatten()
    skip = int(round(seconds * sr))
    trimmed = arr[skip:]
    return trimmed if trimmed.size > 0 else None

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

def _write_wav(path, arr, sr: int) -> None:
    import wave
    arr = np.asarray(arr, dtype=np.float32).flatten()
    pcm = (arr * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())

def _read_wav(path) -> np.ndarray:
    import wave
    with wave.open(str(path), 'rb') as wf:
        assert wf.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        pcm = wf.readframes(wf.getnframes())
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    return arr

def load_ref_audio_cache(root, split: str, task: str, partition: str = 'clean', dbg=None) -> list:
    assert task in ('tts', 'asr'), f"load_ref_audio_cache: unknown task={task!r}"
    root = Path(str(root)) if not isinstance(root, Path) else root
    if split == 'val':
        path = root / 'validation' / f'cache_val_{task}_632.ref_audio.npz'
    else:
        path = root / 'test' / f'cache_test_{partition}_{task}.ref_audio.npz'

    if not path.exists():
        msg = f"[{task} ref-audio] {path} not found -- reference audio will be skipped for now"
        if dbg:
            dbg(msg)
        else:
            print(msg)
        return []

    data = np.load(path, allow_pickle=True)
    return list(data['wavs'])

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
    def __init__(
            self,
            whisper_model: str = 'openai/whisper-medium',
            sr=16_000,
            speaker_checkpoint: str = 'assets/wavlm_large_finetune.pth',
            speech_extractor: str = 'microsoft/wavlm-large',
            text_model: str = 'meta-llama/Llama-3.2-1B',
            statistics_path: str = None,
            e2v_model: str = 'iic/emotion2vec_base',
            e2v_statistics_path: str = 'datasets/test/fsd_ref_stats_500_e2v.npz',
            partition: str = 'clean',
            _dbg_func=None,
            speaker_ref_path: str = 'datasets/test/ref_speaker_embeddings.npz',
            asr_num_workers: int = 4,
        ):
        self._whisper_name = whisper_model
        self._speaker_checkpoint_path = speaker_checkpoint
        self._feature_extractor_name = speech_extractor
        self._text_model_name = text_model
        self._statistics_path = statistics_path
        self._e2v_model_name = e2v_model
        self._e2v_statistics_path = e2v_statistics_path
        self._speaker_ref_path = speaker_ref_path
        self._asr_num_workers = int(asr_num_workers)
        self._sr = sr
        self._asr = None
        self._speaker_model = None
        self._fsd_feature_extractor = None
        self._feature_extractor = None
        self._e2v_model = None
        self._text_extractor = None
        self._text_model = None
        self.reference_statistics = None
        self.e2v_reference_statistics = None
        self.partition = partition
        self._utmos = None
        self._dbg_func = _dbg_func

        self._ref_speaker_embeddings = None

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
                preds = asr(inputs, batch_size=min(64, len(inputs)), num_workers=self._asr_num_workers)
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

    def _ensure_speaker_model(self, device):
        if self._speaker_model is None:
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
            state_dict = torch.load(self._speaker_checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=False)
            model.eval()
            self._speaker_model = model.to(device)
        return self._speaker_model

    def _ensure_ref_speaker_embeddings(self, device):
        if self._ref_speaker_embeddings is None:
            data = np.load(self._speaker_ref_path)
            embeddings = data[f'{self.partition}_embeddings']
            self._ref_speaker_embeddings = torch.from_numpy(embeddings).float()
        return self._ref_speaker_embeddings.to(device)

    def spksim(self, gen_wavs, device) -> float:
        model = self._ensure_speaker_model(device)
        ref_embeddings = self._ensure_ref_speaker_embeddings(device).cpu()

        gen_embeddings = []
        with torch.no_grad():
            for wav in gen_wavs:
                arr = np.asarray(wav, dtype=np.float32).flatten()
                x = torch.from_numpy(arr).unsqueeze(0).to(device)
                emb = model(x)
                gen_embeddings.append(emb.squeeze(0).cpu())
        gen_embeddings = torch.nn.functional.normalize(torch.stack(gen_embeddings, dim=0), dim=-1)

        return (ref_embeddings * gen_embeddings).sum(dim=-1).mean().item()

    def _ensure_feature_extractor(self, device):
        if self._fsd_feature_extractor is None:
            self._fsd_feature_extractor = AutoFeatureExtractor.from_pretrained(
                self._feature_extractor_name
            )
        if self._feature_extractor is None:
            self._feature_extractor = WavLMModel.from_pretrained(
                self._feature_extractor_name
            ).to(device)
            self._feature_extractor.eval()
        return self._fsd_feature_extractor, self._feature_extractor
    
    def _ensure_refernce_statistics(self):
        if self.reference_statistics is None:
            stats = np.load(self._statistics_path)
            self.reference_statistics = (stats['mean'], stats['cov'])
        return self.reference_statistics

    @staticmethod
    def _frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov) -> float:
        mean_diff = ref_mean - gen_mean
        tr, _ = sqrtm(ref_cov @ gen_cov, disp=False)
        if np.iscomplexobj(tr):
            tr = tr.real
        tr = np.trace(ref_cov + gen_cov - 2 * tr)
        return mean_diff @ mean_diff + tr

    def fsd(self, wavs, device, chunk_size: int = 32) -> float:
        extractor, model = self._ensure_feature_extractor(device)
        ref_mean, ref_cov = self._ensure_refernce_statistics()

        wavs = [np.asarray(wav, dtype=np.float32).flatten() for wav in wavs]

        all_embs = []
        for i in range(0, len(wavs), chunk_size):
            chunk = wavs[i : i + chunk_size]
            inputs = extractor(chunk, return_tensors='pt', padding=True, sampling_rate=self._sr)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            all_embs.append(out.hidden_states[6].reshape(-1,1024).cpu().float().numpy())
        embedding = np.concatenate(all_embs, axis=0)

        gen_mean = np.mean(embedding, axis=0)
        gen_cov = np.cov(embedding, rowvar=False)
        return self._frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov)

    def _ensure_e2v_model(self, device):
        if self._e2v_model is None:
            from funasr import AutoModel
            am = AutoModel(model=self._e2v_model_name, device=str(device))
            self._e2v_model = am.model.to(device).eval()
        return self._e2v_model

    def _ensure_e2v_reference_statistics(self):
        if self.e2v_reference_statistics is None:
            stats = np.load(self._e2v_statistics_path)
            self.e2v_reference_statistics = (stats['mean'], stats['cov'])
        return self.e2v_reference_statistics

    @torch.no_grad()
    def _e2v_frame_embeddings(self, wavs, model, device, batch_size: int = 32) -> np.ndarray:
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
        return np.concatenate(out, axis=0)

    def fsd_e2v(self, wavs, device, chunk_size: int = 32) -> float:
        model = self._ensure_e2v_model(device)
        ref_mean, ref_cov = self._ensure_e2v_reference_statistics()

        wavs = [np.asarray(wav, dtype=np.float32).flatten() for wav in wavs]
        embedding = self._e2v_frame_embeddings(wavs, model, device, batch_size=chunk_size)

        gen_mean = np.mean(embedding, axis=0)
        gen_cov = np.cov(embedding, rowvar=False)
        return self._frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov)

    def _ensure_text_model(self, device):
        if self._text_extractor is None:
            self._text_extractor = AutoTokenizer.from_pretrained(self._text_model_name)
            self._text_extractor.pad_token = self._text_extractor.eos_token
        if self._text_model is None:
            self._text_model = AutoModelForCausalLM.from_pretrained(
                self._text_model_name, torch_dtype=torch.float16
            ).to(device).eval()
        return self._text_extractor, self._text_model

    def gen_ppl(self, hyps, device, chunk_size: int = 32) -> float:
        extractor, model = self._ensure_text_model(device)

        all_perplexities: List[float] = []
        for i in range(0, len(hyps), chunk_size):
            chunk = hyps[i : i + chunk_size]
            inputs = extractor(chunk, return_tensors='pt', padding=True).to(model.device)
            labels = inputs['input_ids'].clone()
            labels[inputs['attention_mask'] == 0] = -100

            with torch.no_grad():
                logits = model(**inputs).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            mask = shift_labels != -100
            mean_loss_per_sample = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
            all_perplexities.extend(torch.exp(mean_loss_per_sample).tolist())

        finite = [p for p in all_perplexities if math.isfinite(p)]
        n_dropped = len(all_perplexities) - len(finite)
        if n_dropped:
            self._dbg(
                f'gen_ppl: {n_dropped}/{len(all_perplexities)} samples had non-finite perplexity'
            )
        if not finite:
            return float('nan')
        return sum(finite) / len(finite)

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

    def evaluate_task_extensive(
        self, task:str, gen_texts:List[str], ref_texts: List[str], gen_wavs, device
    ) -> Dict[str, float]:
        metrics = {}
        transcriptions = None

        if task == 'joint':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['Cross-Modal WER'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['Cross-Modal CER'] = self.character_error_rate(gen_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
            metrics['GenPPL-text'] = self.gen_ppl(gen_texts, device)
            metrics['GenPPL-speech'] = self.gen_ppl(transcriptions, device)
            metrics['FSD'] = self.fsd(gen_wavs, device)
        elif task == 'tts':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['ASR-WER'] = self.word_error_rate(ref_texts, transcriptions)
            metrics['ASR-CER'] = self.character_error_rate(ref_texts, transcriptions)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
            metrics['SpkSim'] = self.spksim(gen_wavs, device)
        elif task == 'stt':
            metrics[f'{self.partition}-WER'] = self.word_error_rate(ref_texts, gen_texts)
            metrics[f'{self.partition}-CER'] = self.character_error_rate(ref_texts, gen_texts)
        elif task == 'cont_taste':
            trimmed_wavs = [_trim_prefix_seconds(w, CONT_TRIM_SECONDS, self._sr) for w in gen_wavs]
            transcriptions = self.transcribe(trimmed_wavs, device)
            metrics['UTMOS'] = self.utmos_score(gen_wavs)
        elif task == 'cont_flowslm':
            transcriptions = self.transcribe(gen_wavs, device)
            metrics['WER'] = self.word_error_rate(gen_texts, transcriptions)
            metrics['CER'] = self.character_error_rate(gen_texts, transcriptions)
            metrics['GenPPL-text'] = self.gen_ppl(gen_texts, device)
            metrics['GenPPL-speech'] = self.gen_ppl(transcriptions, device)
            metrics['FSD-wlm'] = self.fsd(gen_wavs, device)
            metrics['FSD-e2v'] = self.fsd_e2v(gen_wavs, device)

        return metrics, transcriptions
 
    def evaluate_stt(self, ref_texts: list[str], gen_wavs=None, transcriptions: list[str] = None, device='cpu'):
        if not transcriptions:
            transcriptions = self.transcribe(gen_wavs, device)
        wer = self.word_error_rate(ref_texts, transcriptions)
        cer = self.character_error_rate(ref_texts, transcriptions)
        return wer, cer

SALMON_JUDGES = [
    "nvidia/speakerverification_en_titanet_large", 
    "ALM/hubert-large-audioset",
    "ALM/wav2vec2-large-audioset"
]

SALMON_JUDGE_PER_TASK = {
    'sentiment_consistency': "nvidia/speakerverification_en_titanet_large",
    'speaker_consistency': "nvidia/speakerverification_en_titanet_large",
    'gender_consistency': "nvidia/speakerverification_en_titanet_large",
    'bg_domain_consistency': "ALM/hubert-large-audioset",
    'bg_all_consistency': "ALM/hubert-large-audioset",
    'rir_consistency': "ALM/wav2vec2-large-audioset"
}

class SALMONEvaluator:
    def __init__(
            self, 
            sr: int = 16_000, 
            max_len: float = 5.0,
            ref_dir: str = 'datasets/continuation',
            _dbg_func=None
        ):
        self._sr = sr
        self._max_len = max_len
        self._dbg_func = _dbg_func
        self.judges = None
        self.embeddings_per_task = None
        self.ref_dir = ref_dir

    def _dbg(self, msg: str):
        if self._dbg_func:
            self._dbg_func(msg)
        else:
            print(msg)

    def _ensure_judges(self, device):
        from utils.judge_models import JudgeModel
        if not self.judges:
            self.judges = {}
            for judge_name in SALMON_JUDGES:
                self.judges[judge_name] = JudgeModel(judge_name, self._sr, device, self._max_len)
        return self.judges
    
    def _ensure_embeddings(self, device):
        if not self.embeddings_per_task:
            self.embeddings_per_task = {}
            for task in SALMON_JUDGE_PER_TASK.keys():
                embeddings_path = f'{self.ref_dir}/judge_embeddings_{task}.npz'
                embeddings = np.load(embeddings_path)
                self.embeddings_per_task[task] = {
                    'pos': torch.from_numpy(embeddings['positive']).to(device),
                    'neg': torch.from_numpy(embeddings['negative']).to(device)
                }
        return self.embeddings_per_task
    
    def evaluate_SALMON(self, gen_wavs_per_task: dict, device):
        metrics = {}

        judges = self._ensure_judges(device)
        ref_embeddings = self._ensure_embeddings(device)

        for task, judge in SALMON_JUDGE_PER_TASK.items():
            gen_embeddings = judges[judge].embed_batch(gen_wavs_per_task[task])
            metrics[task] = judges[judge].score(
                gen_embeddings, ref_embeddings[task]['pos'], ref_embeddings[task]['neg']
            )

        return metrics
