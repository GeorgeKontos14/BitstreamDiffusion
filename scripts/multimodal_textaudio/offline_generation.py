from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Every callback class Trainer.__init__ can construct, besides TextAudioCallback.
_OTHER_CALLBACK_ATTRS = (
    "SigmaDataEstimator",
    "EntropySchedulePlotCallback",
    "OfflineEntropyProfileCallback",
    "VLBBoundCallback",
    "ExternalPPLCallback",
    "MauveCallback",
    "VisualizationCallback",
)


class _NoOpCallback:
    """Stand-in for every non-textaudio callback Trainer.__init__ would
    otherwise construct -- accepts any constructor args, does nothing."""

    def __init__(self, *args, **kwargs):
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to config .py, same as train.py --config.")

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint", default=None,
        help="Path to a checkpoint .pt file, e.g. runs/<exp>/checkpoints/step=000100000.pt.",
    )
    ckpt_group.add_argument(
        "--latest", action="store_true",
        help="Use runs/<experiment>/checkpoints/last.pt instead of a specific --checkpoint.",
    )

    p.add_argument(
        "--eval-name", required=True,
        help="Leaf subfolder name under runs/<experiment>/textaudio_offline/step_<NNNNNNNNN>/.",
    )
    p.add_argument(
        "--entropy-run-dir", default=None,
        help="Dir containing entropy_{pdf,cdf,sigmas,edges}.pt for the entropic "
             "sampler. Defaults to runs/<experiment>/ (whatever the live job most "
             "recently wrote there); pass a snapshot dir to pin an exact state.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from evaluation.utils import load_config
    cfg = load_config(args.config)

    run_dir = REPO_ROOT / "runs" / cfg.experiment

    if args.latest:
        ckpt_path = (run_dir / "checkpoints" / "last.pt").resolve()
    else:
        ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    import train as _train_entry  # for _update_cfg_from_dict, same resume-merge logic train.py uses

    saved_cfg_path = run_dir / "config.json"
    if saved_cfg_path.exists():
        with open(saved_cfg_path, "r", encoding="utf-8") as f:
            saved_cfg_dict = json.load(f)

        _train_entry._update_cfg_from_dict(
            cfg, saved_cfg_dict, skip_sections=("logging", "system", "evaluation", "textaudio"),
        )
        print(f"[offline-gen] merged saved config from {saved_cfg_path} (kept this file's cfg.train.textaudio)")

    from ml_collections import config_dict

    # Non-distributed, single-GPU setup 
    cfg.system = config_dict.ConfigDict()
    cfg.system.distributed = False
    cfg.system.global_rank = 0
    cfg.system.local_rank = 0
    cfg.system.world_size = 1
    cfg._config_path = str(Path(args.config).resolve())

    if getattr(cfg.train, "textaudio", None) is None:
        cfg.train.textaudio = config_dict.ConfigDict()
    cfg.train.textaudio.enabled = True
    cfg.train.textaudio.split = "val"  # exclusively the validation split
    cfg.train.textaudio.entropy_run_dir = args.entropy_run_dir or str(run_dir)

    # ---- SAFETY: never let constructing Trainer touch the live run's files,
    # ---- or construct any callback besides TextAudioCallback ----
    import trainers.trainer as _trainer_mod

    _orig_save_config = _trainer_mod._save_config_to_run_dir
    _trainer_mod._save_config_to_run_dir = lambda cfg, run_dir: None

    _orig_callback_classes = {name: getattr(_trainer_mod, name) for name in _OTHER_CALLBACK_ATTRS}
    for name in _OTHER_CALLBACK_ATTRS:
        setattr(_trainer_mod, name, _NoOpCallback)

    if getattr(cfg, "logging", None) is None:
        cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    if getattr(cfg.logging, "tensorboard", None) is None:
        cfg.logging.tensorboard = config_dict.ConfigDict()
    cfg.logging.tensorboard.enabled = False

    # Checkpoint loading
    cfg.train.init_from = str(ckpt_path)
    cfg.train.init_from_force = True

    try:
        from trainers.trainer import Trainer
        from utils.callbacks.textaudio_generation import TextAudioCallback

        trainer = Trainer(cfg)
    finally:
        _trainer_mod._save_config_to_run_dir = _orig_save_config
        for name, cls in _orig_callback_classes.items():
            setattr(_trainer_mod, name, cls)

    # drop anything that isn't TextAudioCallback
    trainer.callbacks = [c for c in trainer.callbacks if isinstance(c, TextAudioCallback)]

    if trainer.resume_mode != "init_from":
        raise RuntimeError(
            f"expected resume_mode='init_from' from {ckpt_path}, got {trainer.resume_mode!r} "
            f"-- cfg.train.init_from_force should have forced this path unconditionally."
        )

    # init_from is a weights-only load -- it deliberately resets global_step
    # to 0 (see Trainer._init_from_checkpoint_weights_only). Restore the
    # checkpoint's own recorded step purely in-memory (no disk writes) so
    # output metadata/step-based naming reflects the checkpoint actually used.
    ckpt_meta = torch.load(ckpt_path, map_location="meta", weights_only=False, mmap=True)
    trainer.global_step = int(ckpt_meta.get("global_step", 0))
    del ckpt_meta

    print(f"[offline-gen] loaded weights from {ckpt_path} (global_step={trainer.global_step})")

    cb = next(c for c in trainer.callbacks if isinstance(c, TextAudioCallback))
    cb._should_run = lambda epoch, r: True  # force the run; bypass every_k_epochs gating

    save_dir = run_dir / "textaudio_offline" / f"step_{trainer.global_step:09d}" / args.eval_name
    cb.on_epoch_end(trainer, epoch=0, save_dir_override=save_dir)

    print(f"[offline-gen] done. Results under: {save_dir}")


if __name__ == "__main__":
    main()
