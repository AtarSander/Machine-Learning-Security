"""Thin fixed-budget adapter to the preserved third_party/k_arm scanner."""

import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

# The preserved vendor uses a sibling absolute import of K_Arm_Scanner.
sys.path.insert(0, str(Path(__file__).parent / "third_party" / "k_arm"))
from Balanced_Scanner import BalancedScanner


def invert(
    model: nn.Module,
    images: Float[Tensor, "batch channels height width"],
    target: int,
    pattern: Float[Tensor, "1 channels height width"],
    mask: Float[Tensor, "1 height width"],
    cfg: DictConfig,
) -> Tuple[Tensor, Tensor]:
    """One complete B batch per step; preserve the scanner's pre-update capture.

    Input images select global versus source-specific scope. The scanner's
    polygon_global mode disables source stopping heuristics; it does not broaden
    the batch. Each call has independent Adam/cost state, with no bandit selection.
    """
    options = SimpleNamespace(
        **OmegaConf.to_container(cfg, resolve=True),
        step=cfg.steps,
        attack_succ_threshold=cfg.asr,
        batch_size=len(images),
        device=str(images.device),
        channels=images.shape[1],
        input_width=images.shape[2],
        input_height=images.shape[3],
        num_classes=1,
    )
    scanner = BalancedScanner(model, options)
    scanner.capture_preupdate = True
    scanner.measurements = []
    indices = torch.arange(len(images))
    loader = DataLoader(
        TensorDataset(images, indices, indices), batch_size=len(images), shuffle=False
    )
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(
        quiet
    ), contextlib.redirect_stderr(quiet):
        patterns, masks, _, _ = scanner.scanning(
            [target], [loader], 0, pattern, mask, "polygon_global", "forward"
        )
    if masks[0] is None:
        raise RuntimeError(
            f"No qualifying patch for target {target}; no fabricated score."
        )
    return masks[0], patterns[0]
