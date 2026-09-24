"""
checkpoint.py -- saving a run so it can be resumed, or used later
"""

from dataclasses import asdict
from pathlib import Path

import torch

from jabarti_llm.config import ModelConfig


def save_checkpoint(path, model, optimizer=None, scheduler=None, step=0, metrics=None):
    """Write everything needed to continue this run, or to rebuild the model.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "model_config": asdict(model.config),
        "step": step,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    torch.save(payload, path)
    return path
    


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    """Restore into an existing model. Returns the step the run had reached.
    """

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    saved_config = ModelConfig( **checkpoint["model_config"] )

    if saved_config != model.config:
        raise ValueError(
            f"Checkpoint was trained with a different model:\n"
            f"  checkpoint: {saved_config}\n"
            f"  this model: {model.config}\n"
            f"Build the model with the checkpoint's config, or use "
            f"model_config_from_checkpoint()."
        )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["step"]


def model_config_from_checkpoint(path, map_location="cpu"):
    """Read a checkpoint's ModelConfig without building anything.
    """
    
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return ModelConfig( **checkpoint["model_config"] )