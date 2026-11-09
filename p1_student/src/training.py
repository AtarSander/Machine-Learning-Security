"""Final CNN fine-tuning using the saved original-image split and candidate caches."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.calibration import ious
from src.data import (
    COORDS,
    GRID,
    LABEL_FREE,
    SCALES,
    SHAPES,
    annotations_at,
    digest,
    read_image,
    residual_maps,
    write_json,
)
from src.model import CropScorer, native_crop, pair_loss

PILOT = Path("outputs/hypothesis_pilots_v1")
QUALITY = Path("outputs/candidate_quality_v1")


def overlaps(boxes, truth):
    return ious(np.asarray(boxes)[None], np.array(truth)[None])[0]


def sample(ident, a):
    rng = np.random.default_rng(int(hashlib.sha256(("42" + ident).encode()).hexdigest()[:16], 16))
    chosen = []
    counts = {}
    labels = {}
    target = overlaps(GRID, a["box"])
    decoys = [r["box"] for r in a["regions"] if r["kind"] == "decoy"]
    decoy = np.max([overlaps(GRID, b) for b in decoys], axis=0) if decoys else np.zeros(265)

    def take(name, pool, count):
        pool = np.array([int(i) for i in pool if i not in chosen], int)
        picked = rng.choice(pool, min(count, len(pool)), replace=False).tolist()
        chosen.extend(picked)
        counts[name] = len(picked)
        for i in picked:
            labels[i] = name

    if any(a["box"]):
        ranked = np.argsort(-target, kind="stable")
        take("good", ranked[:24], 8)
        take("partial", np.flatnonzero((target > 0) & (target < target.max())), 8)
        take("decoy", np.flatnonzero(decoy > 0), 8)
        take("grid", np.arange(265), 8)
    else:
        take("decoy", np.flatnonzero(decoy > 0), 16)
        take("grid", np.arange(265), 16)
    take("fallback", np.arange(265), 32 - len(chosen))
    assert len(chosen) == len(set(chosen)) == 32
    chosen = sorted(chosen)
    return GRID[chosen], counts, [labels[i] for i in chosen]


def pair_schedule_variable(table, steps=4000):
    rng = np.random.default_rng(42)
    pools = {}
    for ident, g in table.groupby("id", sort=True):
        y = g.eval_iou.to_numpy()
        rows = g.index.to_numpy()
        b = g[COORDS].to_numpy()
        diff = y[:, None] - y[None, :]
        valid = diff > 1e-8
        if y.max() == 0 or not valid.any():
            continue
        lo = np.maximum(b[:, None, :2], b[None, :, :2])
        hi = np.minimum(b[:, None, 2:], b[None, :, 2:])
        intersection = np.maximum(hi - lo, 0).prod(2)
        area = (b[:, 2:] - b[:, :2]).prod(1)
        overlap = intersection / (area[:, None] + area[None, :] - intersection)
        near = np.argwhere((y[:, None] >= 0.3) & (y[None, :] > 0.1) & (overlap >= 0.3) & valid)
        if not len(near):
            near = np.argwhere(valid)
        useful = np.flatnonzero(y >= 0.5)
        if not len(useful):
            useful = np.flatnonzero(y == y.max())
        poor = np.flatnonzero(y <= 0.1)
        if not len(poor):
            poor = np.flatnonzero(y == y.min())
        top = np.argsort(-g.attack_score.to_numpy(), kind="stable")[:10]
        hard = top[np.isin(top, poor)]
        pools[ident] = (rows, near, useful, poor, hard)
    ids = sorted(pools)
    if len(ids) < 8:
        raise ValueError("Need at least eight attacked images with unequal candidate IoUs")
    schedule = np.empty((steps, 8, 2, 2), np.int32)
    source_schedule = []
    for step in range(steps):
        picked = rng.choice(ids, 8, replace=False)
        source_schedule.append(picked)
        for k, ident in enumerate(picked):
            rows, near, good, poor, hard = pools[ident]
            schedule[step, k, 0] = rows[near[rng.integers(len(near))]]
            low = hard if len(hard) and (step + k) % 2 == 0 else poor
            schedule[step, k, 1] = [rows[rng.choice(good)], rows[rng.choice(low)]]
    y = table.eval_iou.to_numpy()
    pairs = schedule.reshape(-1, 2)
    assert (y[pairs[:, 0]] > y[pairs[:, 1]]).all()
    return schedule, np.array(source_schedule)


def split():
    plan = json.loads((PILOT / "study_plan.json").read_text())
    parent = json.loads((QUALITY / "split_manifest.json").read_text())
    train_ids = set(parent["fit_image_ids"]) - set(plan["inner"])
    excluded = set(plan["inner"] + parent["calibration_image_ids"] + parent["holdout_image_ids"])
    if train_ids & excluded or len(train_ids) != 3072:
        raise ValueError("Training split changed or overlaps validation")
    return plan, train_ids


def acquire(args):
    """Compute 32 training candidates per additional image; resume existing records."""
    from src.candidates import annotate
    from src.inference import extractor

    plan, train_ids = split()
    annotations = annotations_at(Path("."))
    args.output.mkdir(parents=True, exist_ok=True)
    records = args.output / "records"
    records.mkdir(exist_ok=True)
    engine = extractor(args, args.output / "raw_cache")
    ids = sorted(train_ids - set(plan["train"]))
    for n, ident in enumerate(ids, 1):
        path = records / f"{ident}.pkl"
        if not path.exists():
            boxes, _, categories = sample(ident, annotations[ident])
            table = annotate(
                engine.extract(read_image(Path("."), ident), boxes),
                ident,
                "fit",
                annotations[ident],
            )
            table["sample_category"] = categories
            temporary = path.with_suffix(".tmp")
            table.to_pickle(temporary)
            temporary.replace(path)
        if n % 64 == 0 or n == len(ids):
            print(f"Features: {n}/{len(ids)}", flush=True)


class Inputs:
    def __init__(self, table, mean, std):
        self.ids = table.id.to_numpy()
        self.boxes = table[COORDS].to_numpy()
        w = table.x_max - table.x_min
        h = table.y_max - table.y_min
        values = np.column_stack([table[LABEL_FREE], w / 64, h / 64, w * h / 4096, w / h]).astype(
            np.float32
        )
        self.x = ((values - mean) / std).astype(np.float32)
        self.residuals = {}
        for ident in sorted(set(self.ids)):
            maps = residual_maps(read_image(Path("."), ident))
            self.residuals[ident] = (
                np.concatenate([maps[s] for s in SCALES], axis=2)
                .transpose(2, 0, 1)
                .astype(np.float32)
            )

    def batch(self, indices):
        crops = np.stack([native_crop(self.residuals[self.ids[i]], self.boxes[i]) for i in indices])
        return torch.from_numpy(self.x[indices]), torch.from_numpy(crops)


def train(args):
    """Warm-start the original CNN; select by full-grid inner-validation IoU."""
    plan, train_ids = split()
    old = pd.read_pickle(QUALITY / "fitting_candidates.pkl")
    new_ids = sorted(train_ids - set(plan["train"]))
    new = pd.concat(
        [pd.read_pickle(args.records / f"{ident}.pkl") for ident in new_ids], ignore_index=True
    )
    if set(new.id) != set(new_ids) or not (new.groupby("id").size() == 32).all():
        raise ValueError("Expected 32 candidates for each additional training image")
    training = (
        pd.concat([old[old.id.isin(plan["train"])], new], ignore_index=True)
        .sort_values(["id", *COORDS])
        .reset_index(drop=True)
    )
    inner = old[old.id.isin(plan["inner"])].sort_values(["id", *COORDS]).reset_index(drop=True)
    if set(training.id) != train_ids or set(inner.id) != set(plan["inner"]):
        raise ValueError("Candidate IDs do not match the fixed split")
    for _, group in inner.groupby("id", sort=False):
        if not np.array_equal(group[COORDS], GRID):
            raise ValueError("Inner validation requires the complete ordered grid")
    conf = json.loads((QUALITY / "config.json").read_text())
    if "attack_score" not in training:
        training["attack_score"] = np.nan
    missing = training.attack_score.isna()
    if missing.any():
        paths = [
            p
            for p in Path("outputs").glob("*/reference/scorers.joblib")
            if digest(p) == conf["frozen_scorer_sha256"]
        ]
        if not paths:
            raise FileNotFoundError("Original frozen scorer needed for hard training pairs")
        frozen = joblib.load(paths[0])["fit_combined"]
        training.loc[missing, "attack_score"] = frozen.predict_proba(
            training.loc[missing, conf["features"]]
        )[:, 1]
    warm_path = PILOT / "models/d_crop_42/selected.pt"
    expected = json.loads((warm_path.parent / "complete.json").read_text())
    if digest(warm_path) != expected["checkpoint_sha256"]:
        raise ValueError("Warm-start checkpoint changed")
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    if warm["columns"] != LABEL_FREE + SHAPES:
        raise ValueError("Unexpected CNN feature schema")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "selected.pt").exists():
        raise FileExistsError("Use a new training output directory")
    tr, va = Inputs(training, warm["mean"], warm["std"]), Inputs(inner, warm["mean"], warm["std"])
    model = CropScorer()
    model.load_state_dict(warm["state"])
    schedule, _ = pair_schedule_variable(training, args.steps)
    target = training.eval_iou.to_numpy(dtype=np.float32)
    ys = inner.eval_iou.to_numpy().reshape(-1, len(GRID))
    attacked = inner.eval_has_attack.to_numpy().reshape(-1, len(GRID))[:, 0].astype(bool)
    best = -1
    history = []

    def validate(step):
        nonlocal best
        model.eval()
        with torch.inference_mode():
            scores = np.concatenate(
                [
                    model(*va.batch(np.arange(i, min(i + 128, len(va.x))))).numpy()
                    for i in range(0, len(va.x), 128)
                ]
            ).reshape(ys.shape)
        metric = float(ys[np.arange(len(ys)), scores.argmax(1)][attacked].mean())
        if step == 0 and abs(metric - expected["inner_selected_iou"]) > 1e-9:
            raise ValueError("Warm-start validation replay failed")
        if metric > best:
            best = metric
            checkpoint = dict(
                warm, state=model.state_dict(), additional_updates=step, warm_start_update=500
            )
            temporary = args.output / "selected.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "selected.pt")
        history.append(dict(step=step, selected_iou=metric))
        write_json(
            args.output / "training.json",
            dict(
                history=history,
                train_ids=sorted(train_ids),
                inner_ids=plan["inner"],
                warm_start_sha256=digest(warm_path),
            ),
        )
        print(f"Update {step}: inner IoU {metric:.6f}", flush=True)

    validate(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for step, pairs in enumerate(schedule, 1):
        model.train()
        indices = pairs.reshape(-1)
        optimizer.zero_grad()
        loss = pair_loss(model(*tr.batch(indices)), torch.from_numpy(target[indices]))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite gradients")
        optimizer.step()
        if step % 250 == 0 or step == args.steps:
            validate(step)
