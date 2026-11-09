"""Fit a rejection threshold on labeled calibration images only."""

import numpy as np


def thresholds(gate, selected_iou, attacked):
    order = np.argsort(-gate, kind="stable")
    g = gate[order]
    delta = np.where(attacked, selected_iou, -1.0)[order]
    ends = np.r_[np.flatnonzero(g[:-1] != g[1:]), len(g) - 1]
    values = np.r_[0, np.cumsum(delta)[ends]]
    ts = np.r_[np.nextafter(g[0], np.inf), g[ends]]
    return float(ts[np.argmax(values)])


def ious(boxes, truth):
    lo = np.maximum(boxes[:, :, :2], truth[:, None, :2])
    hi = np.minimum(boxes[:, :, 2:], truth[:, None, 2:])
    intersection = np.maximum(hi - lo, 0).prod(2)
    ba = (boxes[:, :, 2:] - boxes[:, :, :2]).prod(2)
    ta = (truth[:, 2:] - truth[:, :2]).prod(1)
    return np.divide(
        intersection,
        ba + ta[:, None] - intersection,
        out=np.zeros_like(intersection, dtype=float),
        where=(ba + ta[:, None] - intersection) > 0,
    )


def outcome(iou, attacked, gate, t):
    return np.where(gate >= t, np.where(attacked, iou, 0), ~attacked).astype(float)


def calibrate(args):
    """Choose the gate threshold from aligned calibration predictions and labels."""
    from src.data import GRID, digest, write_json
    from src.inference import cached_data

    scored = cached_data(args.scores)
    reference = cached_data(args.labels)
    if not np.array_equal(scored["ids"], reference["ids"]):
        raise ValueError("Calibration IDs/order mismatch")
    scores, quality = scored["scores"], scored["quality"]
    shape = (len(scored["ids"]), len(GRID))
    if (
        scores.shape != shape
        or quality.shape != shape
        or not np.isfinite(scores).all()
        or not np.isfinite(quality).all()
    ):
        raise ValueError("Invalid calibration scores")
    truth = reference["truth"]
    selected = ious(GRID[scores.argmax(1)][:, None], truth)[:, 0]
    threshold = thresholds(quality.max(1), selected, (truth != 0).any(1))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        dict(
            threshold=threshold,
            checkpoint_sha256=str(scored["checkpoint_sha256"]),
            calibration_sha256=digest(args.scores),
        ),
    )
    print(f"Calibrated threshold: {threshold}")
