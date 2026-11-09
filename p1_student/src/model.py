"""Candidate crop CNN, unchanged preprocessing and pairwise ranking loss."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data import GRID, LABEL_FREE, SCALES, residual_maps


class CropScorer(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(3200, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(64 + dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, crop):
        return self.head(torch.cat([self.conv(crop), x], 1))[:, 0]


def pair_loss(scores, targets):
    s = scores.reshape(-1, 2)
    t = targets.reshape(-1, 2)
    delta = t[:, 0] - t[:, 1]
    if not torch.all(delta > 0):
        raise ValueError("Only ordered unequal-target pairs may enter ranking loss")
    return (delta * F.softplus(-(s[:, 0] - s[:, 1]))).mean()


def native_crop(residual, box):
    """Residuals are 6x64x64, already computed on the full image in raw RGB units."""
    x, y, xx, yy = map(int, box)
    left = (x + xx) // 2 - 20
    top = (y + yy) // 2 - 20
    crop = np.zeros((8, 40, 40), np.float32)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(64, left + 40), min(64, top + 40)
    crop[:6, y0 - top : y1 - top, x0 - left : x1 - left] = residual[:, y0:y1, x0:x1]
    crop[7, y0 - top : y1 - top, x0 - left : x1 - left] = 1
    crop[
        6,
        max(y, top) - top : min(yy, top + 40) - top,
        max(x, left) - left : min(xx, left + 40) - left,
    ] = 1
    return crop


def crop_scores(image, features, checkpoint, model, device):
    columns = LABEL_FREE + ["width_fraction", "height_fraction", "area_fraction", "aspect_ratio"]
    if checkpoint["columns"] != columns:
        raise ValueError("Unexpected crop feature schema")
    values = features[columns].to_numpy(dtype=np.float32)
    values = ((values - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    residual = residual_maps(image)
    residual = (
        np.concatenate([residual[s] for s in SCALES], axis=2).transpose(2, 0, 1).astype(np.float32)
    )
    result = []
    with torch.inference_mode():
        for start in range(0, len(GRID), 128):
            crops = np.stack([native_crop(residual, b) for b in GRID[start : start + 128]])
            result.extend(
                model(
                    torch.from_numpy(values[start : start + 128]).to(device),
                    torch.from_numpy(crops).to(device),
                )
                .cpu()
                .numpy()
            )
    result = np.asarray(result)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite crop scores")
    return result
