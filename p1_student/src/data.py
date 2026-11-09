"""Native image loading, annotation parsing, fixed grid and residual features."""

import csv
import hashlib
import json
import os
import re
from fractions import Fraction

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

COORDS = ["x_min", "y_min", "x_max", "y_max"]
SCALES = (0.7, 1.2)
SIZES = ((14, 28), (28, 14), (16, 24), (24, 16), (20, 20), (24, 24))
LABEL_FREE = [
    f"intervention_{s}_{k}"
    for s in SCALES
    for k in ("predicted_margin_drop", "predicted_probability_change", "probability_l2_distance")
]
SHAPES = ["width_fraction", "height_fraction", "area_fraction", "aspect_ratio"]


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


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_image(data_dir, ident):
    with Image.open(data_dir / "train/images" / (ident + ".png")) as im:
        if im.mode != "RGB" or im.size != (64, 64):
            raise ValueError(f"Invalid image: {ident}")
        return np.asarray(im, dtype=np.float64) / 255


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
    box_valid(result, empty=True)
    return result


def grid_boxes():
    boxes = []
    for w, h in SIZES:
        xs, ys = (
            sorted(set(range(0, 65 - w, 8)) | {64 - w}),
            sorted(set(range(0, 65 - h, 8)) | {64 - h}),
        )
        boxes.extend((x, y, x + w, y + h) for y in ys for x in xs)
    return sorted(set(boxes))


def read_native(path):
    with Image.open(path) as im:
        if im.mode != "RGB" or im.size != (64, 64):
            raise ValueError("Expected native RGB 64x64")
        return np.asarray(im, dtype=np.float64) / 255


def annotations_at(data_dir):
    result = {}
    with (data_dir / "train_labels.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            box = tuple(int(row[k]) for k in COORDS)
            regions, meta = parse_comment(row["comment"], box)
            if row["id"] in result:
                raise ValueError("Duplicate image id")
            result[row["id"]] = dict(box=box, regions=regions, **meta)
    return result


def immutable_json(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Inputs changed: {path}; use a new output directory")
    write_json(path, value)
    return value


def feature_frame(table, columns):
    if (
        len(columns) != 50
        or columns[-6:] != LABEL_FREE
        or not all(c.startswith(("s0.7_", "s1.2_")) for c in columns[:44])
    ):
        raise ValueError("Only the unchanged 44 residual + six response features are allowed")
    result = table[columns].copy()
    width = table.x_max.to_numpy() - table.x_min.to_numpy()
    height = table.y_max.to_numpy() - table.y_min.to_numpy()
    if np.any(width <= 0) or np.any(height <= 0):
        raise ValueError("Nonempty candidate rectangles required")
    for name, values in zip(
        SHAPES, [width / 64, height / 64, width * height / 4096, width / height]
    ):
        result[name] = values
    if not np.isfinite(result).all().all():
        raise ValueError("Nonfinite inference features")
    return result


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


GRID = np.array(grid_boxes(), dtype=np.int64)
