"""Frozen ResNet interventions with per-image feature caching."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from src.data import (
    COORDS,
    SCALES,
    area,
    box_valid,
    digest,
    features,
    iou,
    output_box,
    torch_setup,
    write_json,
)


def replace_region(image, denoised, box):
    x, y, xx, yy = map(int, box)
    box_valid((x, y, xx, yy))
    modified = image.copy()
    modified[y:yy, x:xx] = denoised[y:yy, x:xx]
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[y:yy, x:xx] = False
    assert np.array_equal(modified[outside], image[outside])
    return modified


def margins(logits, indices):
    rows = np.arange(len(logits))
    own = logits[rows, indices]
    others = logits.copy()
    others[rows, indices] = -np.inf
    return own - others.max(axis=1)


class CandidateScorer:
    """Only images and rectangles enter this extractor; no annotations are loaded."""

    def __init__(self, args, cache):
        import importlib.util

        self.torch, self.device = torch_setup(args)
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        model_dir = args.data_dir / "model"
        spec = importlib.util.spec_from_file_location(
            "localization_resnet", model_dir / "resnet.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.resnet18_tin()
        self.model.load_state_dict(
            self.torch.load(model_dir / "resnet18_tin.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model.eval().requires_grad_(False).to(self.device)
        signature = dict(
            weights_sha256=digest(model_dir / "resnet18_tin.pt"),
            implementation_sha256=digest(model_dir / "resnet.py"),
            scales=list(SCALES),
            schema=1,
            torch_version=self.torch.__version__,
            device=self.device,
        )
        if (cache / "signature.json").exists() and json.loads(
            (cache / "signature.json").read_text()
        ) != signature:
            raise ValueError("Candidate cache preprocessing/model changed")
        write_json(cache / "signature.json", signature)
        self.counts = dict(
            original_forward_images=0,
            intervention_forward_images=0,
            forward_batches=0,
            cached_candidates=0,
            new_candidates=0,
        )

    def infer(self, images, original=False):
        result = []
        with self.torch.inference_mode():
            for offset in range(0, len(images), 32):
                values = (
                    np.stack(images[offset : offset + 32]).transpose(0, 3, 1, 2).astype(np.float32)
                )
                z = self.model(self.torch.from_numpy(values).to(self.device)).cpu().numpy()
                assert z.shape == (len(values), 200) and np.isfinite(z).all()
                self.counts["forward_batches"] += 1
                self.counts[
                    "original_forward_images" if original else "intervention_forward_images"
                ] += len(values)
                result.append(z)
        return np.concatenate(result)

    def residual_features(self, maps, boxes):
        return [features(maps, box) for box in boxes]

    def extract(self, image, boxes):
        boxes = [output_box(b) for b in boxes]
        if any(area(b) == 0 for b in boxes) or len(boxes) != len(set(boxes)):
            raise ValueError("Extractor requires nonempty, deduplicated boxes")
        key = hashlib.sha256(image.tobytes()).hexdigest()
        original_file, feature_file = self.cache / (key + ".npz"), self.cache / (key + ".csv")
        if original_file.exists():
            with np.load(original_file) as data:
                z0 = data["logits"]
                smooth = {s: data[f"smooth_{s}"] for s in SCALES}
        else:
            smooth = {s: gaussian_filter(image, sigma=(s, s, 0), mode="reflect") for s in SCALES}
            z0 = self.infer([image], original=True)[0]
            np.savez_compressed(
                original_file, logits=z0, **{f"smooth_{s}": v for s, v in smooth.items()}
            )
        maps = {s: image - v for s, v in smooth.items()}
        cached = pd.read_csv(feature_file) if feature_file.exists() else pd.DataFrame()
        known = set(map(tuple, cached[COORDS].to_numpy())) if len(cached) else set()
        missing = [b for b in boxes if b not in known]
        self.counts["cached_candidates"] += len(boxes) - len(missing)
        self.counts["new_candidates"] += len(missing)
        if missing:
            modified = [replace_region(image, smooth[s], b) for b in missing for s in SCALES]
            logits = self.infer(modified)
            predicted = int(z0.argmax())
            p0 = self.torch.softmax(self.torch.from_numpy(z0), dim=0).numpy()
            probs = self.torch.softmax(self.torch.from_numpy(logits), dim=1).numpy()
            fixed = np.full(len(logits), predicted)
            drops = margins(z0[None], [predicted])[0] - margins(logits, fixed)
            residual_features = self.residual_features(maps, missing)
            records = []
            for i, b in enumerate(missing):
                record = dict(zip(COORDS, b))
                record.update(residual_features[i])
                for j, s in enumerate(SCALES):
                    index = i * len(SCALES) + j
                    record[f"intervention_{s}_predicted_margin_drop"] = float(drops[index])
                    record[f"intervention_{s}_predicted_probability_change"] = float(
                        probs[index, predicted] - p0[predicted]
                    )
                    record[f"intervention_{s}_probability_l2_distance"] = float(
                        np.linalg.norm(probs[index] - p0)
                    )
                records.append(record)
            cached = pd.concat([cached, pd.DataFrame(records)], ignore_index=True)
            temp = feature_file.with_suffix(".tmp")
            cached.to_csv(temp, index=False)
            temp.replace(feature_file)
        result = (
            cached.set_index(COORDS)
            .reindex(pd.MultiIndex.from_tuples(boxes, names=COORDS))
            .reset_index()
        )
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Nonfinite candidate features")
        return result


def annotate(table, ident, phase, annotation):
    result = table.copy()
    result["id"], result["phase"] = ident, phase
    result["image_kind"] = annotation["image_kind"]
    result["attack_family"] = annotation["attack_family"]
    result["jpeg_quality"] = annotation["jpeg_quality"]
    result["eval_iou"] = [
        iou(b, annotation["box"]) for b in result[COORDS].itertuples(index=False, name=None)
    ]
    result["eval_has_attack"] = area(annotation["box"]) > 0
    result["eval_true_area"] = area(annotation["box"])
    result["eval_decoy_types"] = ";".join(
        sorted({r["region_type"] for r in annotation["regions"] if r["kind"] == "decoy"})
    )
    attack = next((r for r in annotation["regions"] if r["kind"] == "adversarial"), {})
    for name in ("epsilon", "l2", "reported_linf"):
        result["eval_attack_" + name] = attack.get(name)
    return result
