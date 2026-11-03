"""Candidates for patch analysis experiments."""

import csv
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers import margins as base_margins
from src.classifiers import replace_region as base_replace_region
from src.data import COORDS as base_COORDS
from src.data import LABEL_FREE, PHASES, area, iou, output_box, stable_rng
from src.data import SCALES as base_SCALES
from src.data import baseline_source as base_baseline_source
from src.data import digest as base_digest
from src.data import discover_split as base_discover_split
from src.data import features as base_features
from src.data import parse_comment as base_parse_comment
from src.data import residual_columns as base_residual_columns
from src.data import torch_setup as base_torch_setup
from src.data import write_json as base_write_json


def fit_logistic(df, columns, target, seed):
    assert all(c.startswith(("s0.7_", "s1.2_", "intervention_")) for c in columns)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    )
    model.fit(df[columns], target)
    return model


def prepare_reference(args, root):
    source = base_baseline_source(args)
    if source is None:
        raise FileNotFoundError("Run the baseline first; expected outputs/full/regions.csv")
    rows = pd.read_csv(source / "regions.csv", dtype={"id": str})
    measured = pd.read_csv(args.reference_dir / "resnet_interventions.csv", dtype={"id": str})
    config = json.loads((args.reference_dir / "resnet_label_free_config.json").read_text())
    columns = base_residual_columns(rows) + LABEL_FREE
    if config["features"] != columns:
        raise ValueError("Winning feature schema differs from the explicit 44+6 allowlist")
    for scale in base_SCALES:
        values = measured[measured.scale == scale].set_index(["id", "region_number"])
        for key in (
            "predicted_margin_drop",
            "predicted_probability_change",
            "probability_l2_distance",
        ):
            rows[f"intervention_{scale}_{key}"] = (
                values[key]
                .reindex(pd.MultiIndex.from_frame(rows[["id", "region_number"]]))
                .to_numpy()
            )
    if not np.isfinite(rows[columns].to_numpy()).all():
        raise ValueError("Missing/nonfinite reference features")
    original = base_discover_split(args)
    for part in ("train", "validation"):
        if set(rows.loc[rows.partition == part, "id"]) != set(original[part + "_image_ids"]):
            raise ValueError("Reference region assignments disagree with the persisted split")
    fit_ids, cal_ids = train_test_split(
        sorted(original["train_image_ids"]), test_size=0.2, random_state=args.seed
    )
    manifest = dict(
        fit_image_ids=sorted(fit_ids),
        calibration_image_ids=sorted(cal_ids),
        holdout_image_ids=sorted(original["validation_image_ids"]),
        original_split=original,
    )
    assert not set(fit_ids) & set(cal_ids)
    assert not (set(fit_ids) | set(cal_ids)) & set(manifest["holdout_image_ids"])
    signature = dict(
        regions_sha256=base_digest(source / "regions.csv"),
        interventions_sha256=base_digest(args.reference_dir / "resnet_interventions.csv"),
        reference_config_sha256=base_digest(args.reference_dir / "resnet_label_free_config.json"),
        columns=columns,
        seed=args.seed,
        manifest=manifest,
    )
    reference = root / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    checkpoint = reference / "scorers.joblib"
    if checkpoint.exists():
        if json.loads((reference / "signature.json").read_text()) != signature:
            raise ValueError("Frozen reference inputs changed; use a new output directory")
        if (
            base_digest(checkpoint)
            != json.loads((reference / "audit.json").read_text())["checkpoint_sha256"]
        ):
            raise ValueError("Frozen reference checkpoint changed")
        models = joblib.load(checkpoint)
    else:
        binary = rows[rows.kind != "background"]
        train = binary[binary.partition == "train"]
        fit = train[train.id.isin(fit_ids)]
        models = {}
        for name, data, features in [
            ("frozen_reference", train, columns),
            ("residual_reference", train, base_residual_columns(rows)),
            ("resnet_reference", train, LABEL_FREE),
            ("fit_combined", fit, columns),
            ("fit_residual", fit, base_residual_columns(rows)),
        ]:
            models[name] = fit_logistic(data, features, data.kind == "adversarial", args.seed)
        joblib.dump(models, checkpoint)
        base_write_json(reference / "signature.json", signature)
        val = binary[binary.partition == "validation"]
        saved = pd.read_csv(
            args.reference_dir / "resnet_label_free_predictions.csv", dtype={"id": str}
        ).set_index(["id", "region_number"])
        expected = saved.probability.reindex(
            pd.MultiIndex.from_frame(val[["id", "region_number"]])
        ).to_numpy()
        actual = models["frozen_reference"].predict_proba(val[columns])[:, 1]
        audit = dict(
            status="Recovered checkpoint from archived training features; original fitted logistic checkpoint was not saved.",
            max_probability_difference_from_archived=float(np.abs(actual - expected).max()),
            mean_probability_difference_from_archived=float(np.abs(actual - expected).mean()),
            validation_regions=len(val),
            explicit_features=columns,
            model_config=config,
            preprocessing="Unchanged full-image residual features and original-prediction-fixed interventions; scalers fit on training only.",
            calibration_policy="Frozen reference has seen original training images: diagnostic only. Deployment uses fit_combined, excluding all calibration images.",
            checkpoint_sha256=base_digest(checkpoint),
        )
        base_write_json(reference / "audit.json", audit)
        print(
            "Recovered reference:",
            audit["max_probability_difference_from_archived"],
            "maximum probability difference",
            flush=True,
        )
    base_write_json(root / "split_manifest.json", manifest)
    with (args.data_dir / "train_labels.csv").open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    annotations = {}
    for label in labels:
        box = tuple(int(label[c]) for c in base_COORDS)
        regions, meta = base_parse_comment(label["comment"], box)
        annotations[label["id"]] = dict(box=box, regions=regions, **meta)
    if set(annotations) != set(rows.id):
        raise ValueError("Reference features do not cover all training images")
    for ident, g in rows.groupby("id", sort=False):
        annotated = g[g.kind != "background"]
        expected = annotations[ident]["regions"]
        if len(annotated) != len(expected) or any(
            tuple(record["box"]) != tuple(saved[c] for c in base_COORDS)
            or record["kind"] != saved.kind
            for record, (_, saved) in zip(expected, annotated.iterrows())
        ):
            raise ValueError(f"Stale reference boxes: {ident}")
    return rows, columns, models, manifest, annotations


def selected_images(manifest, annotations, limit, seed):
    selection = {}
    if limit is not None and limit < 9:
        raise ValueError(
            "Use --limit >= 9 to include three image groups in fit/calibration/holdout"
        )
    for j, phase in enumerate(PHASES):
        ids = manifest[phase + "_image_ids"]
        count = len(ids) if limit is None else min(len(ids), limit // 3 + (j < limit % 3))
        if count == len(ids):
            selection[phase] = ids
            continue
        # Preserve each phase and approximate dataset prevalence; tiny smoke runs include every group.
        nonattack = max(1, count // 10)
        wanted = {"ADV": count - 2 * nonattack, "CLEAN": nonattack, "DECOY": nonattack}
        chosen = []
        for kind, n in wanted.items():
            available = sorted(i for i in ids if annotations[i]["image_kind"] == kind)
            if len(available) < n:
                raise ValueError(f"Insufficient {kind} images in {phase} for stratified subset")
            chosen.extend(
                stable_rng(seed, phase + kind).choice(available, size=n, replace=False).tolist()
            )
        selection[phase] = sorted(chosen)
    return selection


class CandidateScorer:
    """Only images and rectangles enter this extractor; no annotations are loaded."""

    def __init__(self, args, cache):
        import importlib.util

        self.torch, self.device = base_torch_setup(args)
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
            weights_sha256=base_digest(model_dir / "resnet18_tin.pt"),
            implementation_sha256=base_digest(model_dir / "resnet.py"),
            scales=list(base_SCALES),
            schema=1,
            torch_version=self.torch.__version__,
            device=self.device,
        )
        if (cache / "signature.json").exists() and json.loads(
            (cache / "signature.json").read_text()
        ) != signature:
            raise ValueError("Candidate cache preprocessing/model changed")
        base_write_json(cache / "signature.json", signature)
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

    def extract(self, image, boxes):
        boxes = [output_box(b) for b in boxes]
        if any(area(b) == 0 for b in boxes) or len(boxes) != len(set(boxes)):
            raise ValueError("Extractor requires nonempty, deduplicated boxes")
        key = hashlib.sha256(image.tobytes()).hexdigest()
        original_file, feature_file = self.cache / (key + ".npz"), self.cache / (key + ".csv")
        if original_file.exists():
            with np.load(original_file) as data:
                z0 = data["logits"]
                smooth = {s: data[f"smooth_{s}"] for s in base_SCALES}
        else:
            smooth = {
                s: gaussian_filter(image, sigma=(s, s, 0), mode="reflect") for s in base_SCALES
            }
            z0 = self.infer([image], original=True)[0]
            np.savez_compressed(
                original_file, logits=z0, **{f"smooth_{s}": v for s, v in smooth.items()}
            )
        maps = {s: image - v for s, v in smooth.items()}
        cached = pd.read_csv(feature_file) if feature_file.exists() else pd.DataFrame()
        known = set(map(tuple, cached[base_COORDS].to_numpy())) if len(cached) else set()
        missing = [b for b in boxes if b not in known]
        self.counts["cached_candidates"] += len(boxes) - len(missing)
        self.counts["new_candidates"] += len(missing)
        if missing:
            modified = [
                base_replace_region(image, smooth[s], b) for b in missing for s in base_SCALES
            ]
            logits = self.infer(modified)
            predicted = int(z0.argmax())
            p0 = self.torch.softmax(self.torch.from_numpy(z0), dim=0).numpy()
            probs = self.torch.softmax(self.torch.from_numpy(logits), dim=1).numpy()
            fixed = np.full(len(logits), predicted)
            drops = base_margins(z0[None], [predicted])[0] - base_margins(logits, fixed)
            records = []
            for i, b in enumerate(missing):
                record = dict(zip(base_COORDS, b))
                record.update(base_features(maps, b))
                for j, s in enumerate(base_SCALES):
                    index = i * len(base_SCALES) + j
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
            cached.set_index(base_COORDS)
            .reindex(pd.MultiIndex.from_tuples(boxes, names=base_COORDS))
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
        iou(b, annotation["box"]) for b in result[base_COORDS].itertuples(index=False, name=None)
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


def add_scores(table, models, columns):
    for name, model in models.items():
        if name in ("frozen_reference", "fit_combined", "adapted_robustness", "adapted_background"):
            table[name] = model.predict_proba(table[columns])[:, 1]
    return table
