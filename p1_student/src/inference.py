"""Final CNN inference, cached-feature scoring and validated CSV export."""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import (
    COORDS,
    GRID,
    LABEL_FREE,
    SHAPES,
    digest,
    feature_frame,
    immutable_json,
    load,
    read_native,
)
from src.model import CropScorer, crop_scores

FINAL = Path("outputs/more_sources_v1")
QUALITY = Path("outputs/candidate_quality_v1")


def extractor(args, cache):
    from src.candidates import CandidateScorer

    config = SimpleNamespace(
        data_dir=Path("."),
        seed=42,
        threads=args.threads,
        device="cpu",
        crop_resnet_backend=args.backend,
    )
    if args.backend == "torch":
        return CandidateScorer(config, cache)
    from src.candidate_openvino import OpenVINOCandidateScorer

    return OpenVINOCandidateScorer(config, cache)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["columns"] != LABEL_FREE + SHAPES:
        raise ValueError("Unexpected CNN feature schema")
    model = CropScorer().eval()
    model.load_state_dict(checkpoint["state"])
    return checkpoint, model


def validate_submission(table, ids):
    if (
        list(table.columns) != ["id", *COORDS]
        or table.id.tolist() != list(ids)
        or table.id.duplicated().any()
    ):
        raise ValueError("Submission schema/order/IDs mismatch")
    boxes = table[COORDS].to_numpy()
    if not np.isfinite(boxes).all() or not np.equal(boxes, np.floor(boxes)).all():
        raise ValueError("Noninteger output")
    valid = (boxes == 0).all(1) | (
        (boxes[:, :2] >= 0).all(1)
        & (boxes[:, 2:] <= 64).all(1)
        & (boxes[:, :2] < boxes[:, 2:]).all(1)
    )
    if not valid.all():
        raise ValueError("Invalid output rectangle")


def submission(ids, scores, quality, threshold):
    shape = (len(ids), len(GRID))
    if (
        scores.shape != shape
        or quality.shape != shape
        or not np.isfinite(scores).all()
        or not np.isfinite(quality).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("Invalid candidate scores or threshold")
    boxes = GRID[scores.argmax(1)].copy()
    boxes[quality.max(1) < threshold] = 0
    table = pd.DataFrame(boxes, columns=COORDS)
    table.insert(0, "id", ids)
    validate_submission(table, ids)
    return table


def write_submission(table, path):
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    validate_submission(pd.read_csv(temporary, dtype={"id": str}), table.id.tolist())
    temporary.replace(path)
    print(f"{path}: {len(table)} images, {(table[COORDS] == 0).all(axis=1).sum()} NONE", flush=True)


def cached_data(path):
    data = load(path)
    if not np.array_equal(data["boxes"], GRID) or len(set(data["ids"])) != len(data["ids"]):
        raise ValueError("Cached candidate grid or IDs changed")
    return data


def export(args):
    """Replay the final validated submission without model computation."""
    decision = json.loads((FINAL / "decision.json").read_text())
    verification = json.loads((FINAL / "verification.json").read_text())
    for path, sha in [
        (FINAL / "selected.pt", decision["checkpoint_sha256"]),
        (FINAL / "test_scores.npy", verification["test_scoring_sha256"]),
        (Path("outputs/rescue_v1/test_base.npz"), verification["test_cache_sha256"]),
    ]:
        if digest(path) != sha:
            raise ValueError(f"Final artifact changed: {path}")
    data = cached_data(Path("outputs/rescue_v1/test_base.npz"))
    table = submission(
        data["ids"], np.load(FINAL / "test_scores.npy"), data["quality"], decision["threshold"]
    )
    validate_submission(table, pd.read_csv("test_ids.csv", dtype={"id": str}).id.tolist())
    write_submission(table, args.output)


def score(args):
    """Rescore cached ResNet responses with a chosen CNN; no ResNet calls."""
    data = cached_data(args.cache)
    checkpoint, model = load_model(args.checkpoint)
    if data["x"].shape != (len(data["ids"]), len(GRID), 10) or not np.isfinite(data["x"]).all():
        raise ValueError("Invalid cached CNN features")
    scores = []
    for n, ident in enumerate(data["ids"]):
        frame = pd.DataFrame(data["x"][n], columns=LABEL_FREE + SHAPES)
        scores.append(
            crop_scores(read_native(args.images / f"{ident}.png"), frame, checkpoint, model, "cpu")
        )
        if (n + 1) % 250 == 0:
            print(f"CNN: {n + 1}/{len(data['ids'])}", flush=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=data["ids"],
        boxes=GRID,
        scores=np.array(scores),
        quality=data["quality"],
        checkpoint_sha256=digest(args.checkpoint),
    )


def predict(args):
    """Inference from native test PNGs, with resumable feature and score caches."""
    decision = json.loads((FINAL / "decision.json").read_text())
    checkpoint_path = FINAL / "selected.pt"
    for path, sha in [
        (checkpoint_path, decision["checkpoint_sha256"]),
        (QUALITY / "quality.joblib", decision["quality_checkpoint_sha256"]),
    ]:
        if digest(path) != sha:
            raise ValueError(f"Final model changed: {path}")
    config = json.loads((QUALITY / "config.json").read_text())
    for file, key in [
        ("model/resnet.py", "resnet_code_sha256"),
        ("model/resnet18_tin.pt", "weights_sha256"),
    ]:
        if digest(Path(file)) != config["raw_contract"][key]:
            raise ValueError("ResNet changed")
    args.cache.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.ids, dtype={"id": str}).id.tolist()
    immutable_json(
        args.cache / "deployment.json",
        dict(
            cnn=digest(checkpoint_path),
            quality=decision["quality_checkpoint_sha256"],
            threshold=decision["threshold"],
            backend=args.backend,
            ids=ids,
            grid=GRID.tolist(),
        ),
    )
    checkpoint, model = load_model(checkpoint_path)
    quality_model = joblib.load(QUALITY / "quality.joblib")
    engine = extractor(args, args.cache / "raw")
    scores, qualities = [], []
    for n, ident in enumerate(ids, 1):
        path = args.images / f"{ident}.png"
        saved = args.cache / f"{ident}.npz"
        sha = digest(path)
        if saved.exists():
            record = load(saved)
            if str(record["image_sha256"]) != sha:
                raise ValueError("Cached image changed")
        else:
            image = read_native(path)
            frame = feature_frame(engine.extract(image, GRID), config["features"])
            record = dict(
                scores=crop_scores(image, frame, checkpoint, model, "cpu"),
                quality=np.clip(quality_model.predict(frame), 0, 1),
                image_sha256=sha,
            )
            np.savez_compressed(saved, **record)
        scores.append(record["scores"])
        qualities.append(record["quality"])
        if n % 25 == 0 or n == len(ids):
            print(f"Prediction: {n}/{len(ids)}", flush=True)
    write_submission(
        submission(ids, np.array(scores), np.array(qualities), decision["threshold"]), args.output
    )


def submit(args):
    """Export newly scored candidates with their separately calibrated threshold."""
    data = cached_data(args.scores)
    calibration = json.loads(args.calibration.read_text())
    if str(data["checkpoint_sha256"]) != calibration["checkpoint_sha256"]:
        raise ValueError("Threshold belongs to a different CNN")
    table = submission(data["ids"], data["scores"], data["quality"], calibration["threshold"])
    validate_submission(table, pd.read_csv(args.ids, dtype={"id": str}).id.tolist())
    write_submission(table, args.output)
