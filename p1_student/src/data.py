"""Data for patch analysis experiments."""

import hashlib
import itertools
import json
import os
import re
from fractions import Fraction

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle as Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
from matplotlib import pyplot as _pyplot

plt = _pyplot

COORDS = ["x_min", "y_min", "x_max", "y_max"]


SCALES = (0.7, 1.2)


COLORS = {"adversarial": "red", "decoy": "cyan", "background": "lime"}


def box_valid(box, empty=False):
    if empty and tuple(box) == (0, 0, 0, 0):
        return
    x, y, xx, yy = box
    if not (0 <= x < xx <= 64 and 0 <= y < yy <= 64):
        raise ValueError(f"invalid box: {box}")


def parameters(text):
    result = dict(epsilon=None, l2=None, reported_linf=None, true_class=None, target_class=None)
    names = dict(
        eps="epsilon", l2="l2", linf="reported_linf", true="true_class", target="target_class"
    )
    for token in text.split():
        key, value = token.split("=", 1)
        if key not in names or result[names[key]] is not None:
            raise ValueError(f"unsupported or repeated parameter: {token}")
        if key in ("true", "target"):
            result[names[key]] = value
        else:
            parts = value.split("/")
            if len(parts) > 2:
                raise ValueError(f"invalid numeric value: {value}")
            result[names[key]] = float(
                Fraction(parts[0]) / (Fraction(parts[1]) if len(parts) == 2 else 1)
            )
        if key not in ("true", "target") and result[names[key]] < 0:
            raise ValueError(f"negative norm: {token}")
    return result


def parse_comment(comment, box):
    box_valid(box, empty=True)
    parts = [p.strip() for p in comment.split(";")]
    head = parts.pop(0)
    regions, jpeg, expected = [], None, 0
    family = "none"
    if head.startswith("ADV "):
        _, family, params = head.split(" ", 2)
        box_valid(box)
        regions.append(dict(kind="adversarial", region_type=family, box=box, **parameters(params)))
    elif head not in ("CLEAN", "DECOY") or tuple(box) != (0, 0, 0, 0):
        raise ValueError(f"unsupported header/box: {head}")
    seen = set()
    for part in parts:
        key = part.split("=", 1)[0] if part.startswith("jpeg=") else "decoys"
        if key in seen:
            raise ValueError(f"duplicate section: {part}")
        seen.add(key)
        if re.fullmatch(r"jpeg=\d+", part):
            jpeg = int(part[5:])
            if not 1 <= jpeg <= 100:
                raise ValueError("invalid JPEG quality")
        else:
            match = re.fullmatch(r"decoys: (\d+)(?: (.*))?", part)
            if not match:
                raise ValueError(f"unsupported section: {part}")
            expected = int(match[1])
            entries = re.split(r"(?<=\]),\s*", match[2] or "") if match[2] else []
            if len(entries) != expected:
                raise ValueError(f"decoy count mismatch: expected {expected}, found {len(entries)}")
            for entry in entries:
                m = re.fullmatch(r"(\w+) (.*?) at \[(\d+),(\d+),(\d+),(\d+)\]", entry)
                if not m:
                    raise ValueError(f"unsupported decoy: {entry}")
                b = tuple(map(int, m.groups()[2:]))
                box_valid(b)
                regions.append(dict(kind="decoy", region_type=m[1], box=b, **parameters(m[2])))
    if head == "CLEAN" and expected or head == "DECOY" and not expected:
        raise ValueError("header/decoy count mismatch")
    return regions, dict(
        attack_family=family,
        jpeg_quality=jpeg,
        image_kind=head.split()[0],
        expected_decoys=expected,
    )


def correlation(a, b):
    a, b = a.ravel(), b.ravel()
    if a.size < 2:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-15 else 0.0


def residual_maps(image):
    return {s: image - gaussian_filter(image, sigma=(s, s, 0), mode="reflect") for s in SCALES}


def features(maps, box):
    x, y, xx, yy = box
    out = {}
    for s, full in maps.items():
        r = full[y:yy, x:xx]
        prefix = f"s{s}_"
        out[prefix + "rms_all"] = float(np.sqrt(np.mean(r**2)))
        for c, name in enumerate("rgb"):
            v = r[:, :, c]
            values = dict(
                mean=v.mean(),
                std=v.std(),
                abs=np.abs(v).mean(),
                rms=np.sqrt((v * v).mean()),
                horizontal=correlation(v[:, :-1], v[:, 1:]),
                vertical=correlation(v[:-1], v[1:]),
            )
            out.update({prefix + name + "_" + k: float(v) for k, v in values.items()})
        for a, b in ((0, 1), (0, 2), (1, 2)):
            out[prefix + "corr_" + "rgb"[a] + "rgb"[b]] = correlation(r[:, :, a], r[:, :, b])
    if not np.isfinite(list(out.values())).all():
        raise ValueError("nonfinite features")
    return out


def residual_columns(df):
    columns = [c for c in df if c.startswith(("s0.7_", "s1.2_"))]
    if len(columns) != 44:
        raise ValueError(f"Expected 44 unchanged residual features, got {len(columns)}")
    return columns


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def baseline_source(args):
    if args.baseline_dir and not (args.baseline_dir / "regions.csv").exists():
        raise FileNotFoundError(f"No regions.csv in {args.baseline_dir}")
    candidates = (
        [args.baseline_dir]
        if args.baseline_dir
        else [args.output_dir, args.data_dir / "outputs/full", args.data_dir / "outputs"]
    )
    return next((p for p in candidates if p and (p / "regions.csv").exists()), None)


def discover_split(args):
    source = baseline_source(args)
    if source is None:
        return None
    if (source / "split_manifest.json").exists():
        manifest = json.loads((source / "split_manifest.json").read_text())
    else:
        saved = pd.read_csv(source / "regions.csv", dtype={"id": str})
        if saved.groupby("id").partition.nunique().max() != 1:
            raise ValueError("Conflicting baseline image assignments")
        manifest = {
            part + "_image_ids": sorted(saved.loc[saved.partition == part, "id"].unique().tolist())
            for part in ("train", "validation")
        }
    if set(manifest["train_image_ids"]) & set(manifest["validation_image_ids"]):
        raise ValueError("Train/validation image leakage in persisted manifest")
    return manifest


def read_image(data_dir, ident):
    with Image.open(data_dir / "train/images" / (ident + ".png")) as im:
        if im.mode != "RGB" or im.size != (64, 64):
            raise ValueError(f"Invalid image: {ident}")
        return np.asarray(im, dtype=np.float64) / 255


def control_box(boxes, dimensions, rng):
    # Exhaust all observed dimensions before declaring placement impossible.
    for i in rng.permutation(len(dimensions)):
        w, h = dimensions[i]
        ys, xs = np.mgrid[: 65 - h, : 65 - w]
        valid = np.ones(xs.shape, dtype=bool)
        for x, y, xx, yy in boxes:
            valid &= (xs + w <= x) | (xs >= xx) | (ys + h <= y) | (ys >= yy)
        positions = np.argwhere(valid)
        if len(positions):
            y, x = positions[rng.integers(len(positions))]
            return (int(x), int(y), int(x + w), int(y + h))
    return None


def torch_setup(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    if args.device == "cpu":
        device = "cpu"
    elif args.device in ("auto", "cuda") and torch.cuda.is_available():
        device = "cuda"
    elif args.device in ("auto", "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch, device


SIZES = ((14, 28), (28, 14), (16, 24), (24, 16), (20, 20), (24, 24))


PHASES = ("fit", "calibration", "holdout")


LABEL_FREE = [
    f"intervention_{s}_{k}"
    for s in SCALES
    for k in ("predicted_margin_drop", "predicted_probability_change", "probability_l2_distance")
]


def stable_rng(seed, key):
    number = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, number]))


def area(box):
    x, y, xx, yy = box
    return max(0, xx - x) * max(0, yy - y)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = area(a) + area(b) - intersection
    return float(intersection / union) if union else 0.0


def output_box(box):
    if len(box) != 4 or any(not np.isfinite(v) or int(v) != v for v in box):
        raise ValueError(f"Box must have four integer coordinates: {box}")
    result = tuple(int(v) for v in box)
    base_box_valid(result, empty=True)
    return result


def official_score(predicted, truth):
    predicted, truth = output_box(predicted), output_box(truth)
    if truth == (0, 0, 0, 0):
        return float(predicted == (0, 0, 0, 0))
    return iou(predicted, truth)


def grid_boxes():
    boxes = []
    for w, h in SIZES:
        xs, ys = (
            sorted(set(range(0, 65 - w, 8)) | {64 - w}),
            sorted(set(range(0, 65 - h, 8)) | {64 - h}),
        )
        boxes.extend((x, y, x + w, y + h) for y in ys for x in xs)
    return sorted(set(boxes))


def variants(box, budget, rng):
    x, y, xx, yy = box
    cx, cy = (x + xx) / 2, (y + yy) / 2
    offsets = (0, -2, 2, -4, 4, -8, 8)
    choices = list(itertools.product(offsets, offsets, (0.75, 1.0, 1.25)))
    result = []
    for j in rng.permutation(len(choices)):
        dx, dy, scale = choices[j]
        w, h = max(1, round((xx - x) * scale)), max(1, round((yy - y) * scale))
        left, top = round(cx + dx - w / 2), round(cy + dy - h / 2)
        candidate = (max(0, left), max(0, top), min(64, left + w), min(64, top + h))
        if area(candidate) and candidate != tuple(box) and candidate not in result:
            output_box(candidate)
            result.append(candidate)
        if len(result) == budget:
            break
    return result


def shortlist(boxes, scores, budget):
    # Fixed geometric diversity; no annotations or numerical coordinate predictors.
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], boxes[i]))
    chosen = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < 0.7 for j in chosen):
            chosen.append(i)
        if len(chosen) == budget:
            break
    if len(chosen) < min(budget, len(boxes)):
        chosen.extend(i for i in order if i not in chosen)
    return [boxes[i] for i in chosen[:budget]]


def image_proposals(image, residual_model, budget):
    """Annotation-free candidate generation, also used by standalone inference."""
    maps = base_residual_maps(image)
    boxes = grid_boxes()
    table = pd.DataFrame([base_features(maps, b) for b in boxes])
    scores = residual_model.predict_proba(table)[:, 1]
    return boxes, shortlist(boxes, scores, budget)


def background_boxes(image, forbidden, budget, rng):
    eligible = [b for b in grid_boxes() if all(iou(b, other) == 0 for other in forbidden)]
    gx, gy = np.gradient(image, axis=(1, 0))
    texture = np.mean(gx * gx + gy * gy, axis=2)
    strength = [float(texture[y:yy, x:xx].mean()) for x, y, xx, yy in eligible]
    hard = shortlist(eligible, strength, (budget + 1) // 2)
    remaining = [b for b in eligible if b not in hard]
    random_boxes = [remaining[i] for i in rng.permutation(len(remaining))[: budget - len(hard)]]
    result = []
    for i in range(max(len(hard), len(random_boxes))):
        if i < len(random_boxes):
            result.append((random_boxes[i], "ordinary_background"))
        if i < len(hard):
            result.append((hard[i], "high_texture_background"))
    return result


base_SCALES = SCALES
base_box_valid = box_valid
base_features = features
base_residual_maps = residual_maps


def read_scored_candidates(path):
    # Hex IDs can look numeric/scientific; keep IDs and near-one
    # probabilities exact so resuming cannot merge image groups.
    return pd.read_csv(path, dtype={"id": str}, float_precision="round_trip")
