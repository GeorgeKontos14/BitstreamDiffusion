from __future__ import annotations

import argparse

import importlib.util
import json
import sys
from pathlib import Path

import torch
from ml_collections import config_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/textaudio/mls_632.py')
    args = parser.parse_args()
    config_path = args.config

    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from trainers.trainer import Trainer
    from utils.callbacks.entropy_schedule_plot import EntropySchedulePlotCallback
    import train as _train_entry

    # Non-distributed setup
    cfg.system = config_dict.ConfigDict()
    cfg.system.distributed = False
    cfg.system.global_rank = 0
    cfg.system.local_rank = 0
    cfg.system.world_size = 1  

    run_dir = Path('runs') / cfg.experiment
    saved_cfg_path = run_dir / 'config.json'
    last_ckpt = run_dir / 'checkpoints' / 'last.pt'
    if last_ckpt.exists() and saved_cfg_path.exists():
        with open(saved_cfg_path, 'r') as f:
            saved_cfg_dict = json.load(f)
        _train_entry._update_cfg_from_dict(cfg, saved_cfg_dict)
        print(f"[regen] merged saved config from {saved_cfg_path}")
    cfg._config_path = str(Path(config_path).resolve())

    raw_ckpt_meta = torch.load(last_ckpt, map_location='meta', weights_only=False, mmap=True)
    current_epoch = int(raw_ckpt_meta.get('epoch', -1))
    del raw_ckpt_meta        

    trainer = Trainer(cfg)
    print(f'[regen] resumed: resume_mode={trainer.resume_mode!r}, global_step={trainer.global_step}')
    if trainer.resume_mode != 'resume':
        raise RuntimeError(
            f"expected resume_mode='resume' from {last_ckpt}, "
            f"got {trainer.resume_mode!r} -- checkpoint missing?"
        )
    print(f'[regen] entropy tables ready: {getattr(trainer, "_entropy_ready", False)}')
    print(f'[regen] plotting for last.pt\'s own epoch={current_epoch} (trainer.start_epoch={trainer.start_epoch})')

    cb = next(c for c in trainer.callbacks if isinstance(c, EntropySchedulePlotCallback))
    cb.every_k_epochs = 1  # force it to run now regardless of the every-5-epochs gate
    cb.on_epoch_end(trainer, epoch=current_epoch)

    out_path = Path(trainer.run_dir) / 'entropy_plots' / f'entropy_epoch{current_epoch + 1:04d}.png'
    print(f'[regen] done -- {out_path} ({"exists" if out_path.exists() else "NOT FOUND -- check for errors above"})')


if __name__ == '__main__':
    main()