"""Noise model for patch analysis experiments."""

import io

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.checks import localization_checks as loc_self_test
from src.data import SCALES as base_SCALES
from src.data import SIZES as loc_SIZES
from src.data import area as loc_area
from src.data import box_valid as base_box_valid
from src.data import grid_boxes as loc_grid_boxes
from src.data import iou as loc_iou
from src.data import output_box as loc_output_box
from src.data import read_scored_candidates
from src.data import residual_maps as base_residual_maps
from src.data import torch_setup as base_torch_setup
from src.metrics import selected_rows as loc_selected_rows

PREPROCESS = dict(
    channels="RGB, signed RGB residual sigma=0.7, signed RGB residual sigma=1.2",
    rgb_scale=1.0,
    residual_scale=10.0,
    gaussian_mode="reflect",
    gaussian_sigmas=list(base_SCALES),
    resolution=[64, 64],
    augmentation=False,
)


DECODER = dict(
    size_bank=loc_SIZES,
    stride=8,
    score="mean probability inside rectangle",
    suppress_iou_greater_than=0.7,
    primary_k=16,
    tie_break="lexicographic coordinates",
)


GRID = np.asarray(loc_grid_boxes(), dtype=int)


INTER = np.maximum(
    0,
    np.minimum(GRID[:, None, 2:], GRID[None, :, 2:])
    - np.maximum(GRID[:, None, :2], GRID[None, :, :2]),
).prod(2)


AREAS = (GRID[:, 2:] - GRID[:, :2]).prod(1)


GRID_IOU = INTER / (AREAS[:, None] + AREAS[None, :] - INTER)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU()
            )

        self.e1, self.e2, self.e3, self.e4 = (
            block(9, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        self.d3, self.d2, self.d1 = block(192, 64), block(96, 32), block(48, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.e4(F.max_pool2d(c, 2))

        def up(z, skip, block):
            return block(
                torch.cat(
                    [
                        F.interpolate(
                            z, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ],
                    1,
                )
            )

        return self.head(up(up(up(d, c, self.d3), b, self.d2), a, self.d1))


def channels(image):
    if image.shape != (64, 64, 3) or not np.isfinite(image).all():
        raise ValueError("Expected finite native 64x64 RGB pixels")
    maps = base_residual_maps(image)
    return (
        np.concatenate([image] + [maps[s] * 10 for s in base_SCALES], axis=2)
        .transpose(2, 0, 1)
        .astype(np.float32)
    )


def union_mask(annotation):
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    for r in annotation["regions"]:
        x, y, xx, yy = loc_output_box(r["box"])
        base_box_valid((x, y, xx, yy))
        mask[0, y:yy, x:xx] = 1
    return mask


def loss_parts(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = (
        1
        - (
            (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        ).mean()
    )
    return bce + dice, bce, dice


def predict(model, x, device, batch_size):
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            result.append(
                model(x[start : start + batch_size].to(device)).sigmoid().cpu().numpy()[:, 0]
            )
    return np.concatenate(result)


def window_scores(probability):
    summed = np.pad(probability.cumsum(0, dtype=np.float64).cumsum(1), ((1, 0), (1, 0)))
    x, y, xx, yy = GRID.T
    return (summed[yy, xx] - summed[y, xx] - summed[yy, x] + summed[y, x]) / AREAS


def decode_scores(scores, k=16):
    order = np.argsort(-np.asarray(scores), kind="stable")
    selected = []
    for index in order:
        if not selected or np.all(GRID_IOU[index, selected] <= 0.7):
            selected.append(int(index))
        if len(selected) == k:
            break
    return selected


def decode(probability, k=16):
    scores = window_scores(probability)
    indices = decode_scores(scores, k)
    return [tuple(GRID[i]) for i in indices], scores[indices]


def max_overlap(boxes, truth):
    return max((loc_iou(b, truth) for b in boxes), default=0.0)


def coverage(probabilities, ids, annotations, k=16):
    attack, decoy = [], []
    for probability, ident in zip(probabilities, ids):
        boxes, _ = decode(probability, k)
        annotation = annotations[ident]
        if loc_area(annotation["box"]):
            attack.append(max_overlap(boxes, annotation["box"]))
        decoy.extend(
            max_overlap(boxes, r["box"]) for r in annotation["regions"] if r["kind"] == "decoy"
        )
    return dict(attack=summarize_overlap(attack), decoy=summarize_overlap(decoy))


def summarize_overlap(values):
    return dict(
        count=len(values),
        mean_best_iou=float(np.mean(values)) if len(values) else None,
        recall={
            str(t): float(np.mean(np.asarray(values) >= t)) if len(values) else None
            for t in (0.3, 0.5, 0.7)
        },
    )


def mask_metrics(probabilities, target):
    target = target[:, 0].numpy().astype(bool)
    pred = probabilities >= 0.5
    intersection = (pred & target).sum((1, 2))
    union = (pred | target).sum((1, 2))
    per = np.divide(intersection, union, out=np.ones(len(union), float), where=union != 0)
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    bce = float(-(target * np.log(p) + (~target) * np.log1p(-p)).mean())
    dice = float(
        (
            1
            - (2 * (probabilities * target).sum((1, 2)) + 1)
            / (probabilities.sum((1, 2)) + target.sum((1, 2)) + 1)
        ).mean()
    )
    nonempty = target.any((1, 2))
    return dict(
        bce=bce,
        dice_loss=dice,
        total_loss=bce + dice,
        mean_mask_iou=float(per.mean()),
        nonempty_mask_iou=float(per[nonempty].mean()) if nonempty.any() else None,
        clean_count=int((~nonempty).sum()),
        clean_predicted_pixel_fraction=float(pred[~nonempty].mean()) if (~nonempty).any() else None,
    )


def self_test(args):
    base_torch_setup(args)
    loc_self_test()
    assert len(GRID) == 265 and not np.any(GRID_IOU == 0.7)
    regions = [dict(box=(60, 61, 64, 64), kind="adversarial"), dict(box=(0, 0, 2, 3), kind="decoy")]
    assert union_mask(dict(regions=regions)).sum() == 18
    assert union_mask(dict(regions=[regions[1]])).sum() == 6
    assert union_mask(dict(regions=[])).sum() == 0
    model = UNet()
    x = torch.zeros(2, 9, 64, 64)
    z = model(x)
    assert z.shape == (2, 1, 64, 64)
    loss, _, _ = loss_parts(z, torch.zeros_like(z))
    assert torch.isfinite(loss)
    loss.backward()
    p = np.zeros((64, 64), np.float32)
    p[40:64, 40:64] = 1
    boxes, _ = decode(p)
    assert len(boxes) == len(set(boxes)) == 16 and boxes == decode(p)[0]
    scores = window_scores(p)
    assert np.allclose(scores, [p[y:yy, x:xx].mean() for x, y, xx, yy in GRID])
    assert np.allclose(channels(np.zeros((64, 64, 3))), 0)
    assert all(loc_iou(a, b) <= 0.7 for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    fixture = pd.DataFrame(
        dict(id=["000000000001", "000000001e00"], attack_score=[np.nextafter(1.0, 0), 1.0])
    )
    restored = read_scored_candidates(io.StringIO(fixture.to_csv(index=False)))
    assert restored.equals(fixture) and len(loc_selected_rows(restored, "attack_score")) == 2
    print(
        "Union masks, clean/decoy-only targets, boundary pixels, U-Net shape/empty loss and deterministic decoder checks passed.",
        flush=True,
    )
