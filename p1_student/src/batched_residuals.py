"""Vectorized evaluation of the existing residual statistics, grouped by box shape."""

from collections import defaultdict

import numpy as np


def correlations(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    if a.shape[1] < 2:
        return np.zeros(len(a))
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        np.sum(a * b, axis=1), denominator, out=np.zeros(len(a)), where=denominator > 1e-15
    )


def features_batch(maps, boxes):
    groups = defaultdict(list)
    result = [{} for _ in boxes]
    for i, (x, y, xx, yy) in enumerate(boxes):
        groups[(xx - x, yy - y)].append(i)
    for indices in groups.values():
        for scale, full in maps.items():
            crops = np.stack([full[b[1] : b[3], b[0] : b[2]] for b in (boxes[i] for i in indices)])
            prefix = f"s{scale}_"
            values = {prefix + "rms_all": np.sqrt(np.mean(crops**2, axis=(1, 2, 3)))}
            for channel, name in enumerate("rgb"):
                v = crops[:, :, :, channel]
                stats = dict(
                    mean=v.mean(axis=(1, 2)),
                    std=v.std(axis=(1, 2)),
                    abs=np.abs(v).mean(axis=(1, 2)),
                    rms=np.sqrt((v * v).mean(axis=(1, 2))),
                    horizontal=correlations(v[:, :, :-1], v[:, :, 1:]),
                    vertical=correlations(v[:, :-1], v[:, 1:]),
                )
                values.update({prefix + name + "_" + k: value for k, value in stats.items()})
            for a, b in ((0, 1), (0, 2), (1, 2)):
                values[prefix + "corr_" + "rgb"[a] + "rgb"[b]] = correlations(
                    crops[:, :, :, a], crops[:, :, :, b]
                )
            for name, column in values.items():
                if not np.isfinite(column).all():
                    raise ValueError("Nonfinite batched residual features")
                for i, value in zip(indices, column):
                    result[i][name] = float(value)
    return result
