import torch
from ml_collections import config_dict

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