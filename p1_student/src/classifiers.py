"""Classifiers for patch analysis experiments."""

import importlib.util
import json
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import (
    SCALES,
    box_valid,
    digest,
    read_image,
    residual_maps,
    torch_setup,
    write_json,
)
from src.metrics import experiment_metrics


def save_predictions(name, df, probabilities, configuration, seconds, output):
    heldout = df[(df.partition == "validation") & (df.kind != "background")].copy()
    if len(probabilities) != len(heldout) or not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid predictions for {name}")
    heldout["probability"] = probabilities
    metrics = experiment_metrics(heldout, "probability")
    metrics.update(configuration=configuration, runtime_seconds=seconds)
    write_json(output / f"{name}_metrics.json", metrics)
    write_json(output / f"{name}_config.json", configuration)
    heldout[["id", "region_number", "kind", "partition", "probability"]].to_csv(
        output / f"{name}_predictions.csv", index=False
    )
    print(f"{name}: {json.dumps(metrics['overall'])}", flush=True)
    return metrics


def tabular_experiment(name, df, columns, args, output):
    start = time.perf_counter()
    train = df[(df.partition == "train") & (df.kind != "background")]
    val = df[(df.partition == "validation") & (df.kind != "background")]
    if name == "boosting":
        config = dict(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.seed,
        )
        model = HistGradientBoostingClassifier(**config)
    else:
        config = dict(C=1.0, max_iter=2000, random_state=args.seed)
        model = make_pipeline(StandardScaler(), LogisticRegression(**config))
    model.fit(train[columns], train.kind == "adversarial")
    probabilities = model.predict_proba(val[columns])[:, 1]
    config.update(
        features=columns, estimator=type(model).__name__, training_images=int(train.id.nunique())
    )
    return save_predictions(name, df, probabilities, config, time.perf_counter() - start, output)


def make_cnn(torch):
    class ResidualCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Conv2d(a, b, 3, padding=1) for a, b in ((6, 16), (16, 24), (24, 32))]
            )
            self.output = torch.nn.Linear(32, 1)

        def forward(self, x, mask):
            # Mask after every convolution so padded activations cannot feed back.
            for conv in self.blocks:
                x = torch.relu(conv(x)) * mask
            pooled = x.sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
            return self.output(pooled).squeeze(1)

    return ResidualCNN()


def cnn_experiment(df, args, output):
    start = time.perf_counter()
    torch, device = torch_setup(args)
    binary = df[df.kind != "background"].copy()
    crops = []
    previous_id, full_maps = None, None
    for row in binary.itertuples():
        if row.id != previous_id:
            full_maps = np.concatenate(
                list(residual_maps(read_image(args.data_dir, row.id)).values()), axis=2
            )
            previous_id = row.id
        x = full_maps[row.y_min : row.y_max, row.x_min : row.x_max]
        crops.append(torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32)))
    train_indices = np.flatnonzero((binary.partition == "train").to_numpy())
    val_indices = np.flatnonzero((binary.partition == "validation").to_numpy())

    # One fixed unit conversion, not a learned transform or per-crop normalization.
    def batch(indices):
        h = max(crops[i].shape[1] for i in indices)
        w = max(crops[i].shape[2] for i in indices)
        x = torch.zeros(len(indices), 6, h, w)
        mask = torch.zeros(len(indices), 1, h, w)
        for j, i in enumerate(indices):
            c = crops[i]
            x[j, :, : c.shape[1], : c.shape[2]] = c * 255
            mask[j, :, : c.shape[1], : c.shape[2]] = 1
        return x.to(device), mask.to(device)

    targets = torch.tensor(
        (binary.kind == "adversarial").to_numpy(), dtype=torch.float32, device=device
    )
    model = make_cnn(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_indices)
        losses = []
        for offset in range(0, len(order), 64):
            indices = order[offset : offset + 64]
            x, mask = batch(indices)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, mask), targets[indices])
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), len(indices)))
        mean_loss = sum(v * n for v, n in losses) / len(order)
        history.append(dict(epoch=epoch + 1, train_loss=mean_loss))
        print(f"CNN epoch {epoch + 1}/{args.epochs}: training loss={mean_loss:.5f}", flush=True)
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for offset in range(0, len(val_indices), 64):
            x, mask = batch(val_indices[offset : offset + 64])
            probabilities.extend(torch.sigmoid(model(x, mask)).cpu().tolist())
    torch.save(model.state_dict(), output / "cnn_weights.pt")
    write_json(output / "cnn_training.json", history)
    config = dict(
        architecture="Conv3x3 6->16->24->32, ReLU+mask after each, masked global mean, Linear32->1",
        parameters=sum(p.numel() for p in model.parameters()),
        optimizer="AdamW",
        lr=0.001,
        weight_decay=0.0001,
        batch_size=64,
        epochs=args.epochs,
        input_units="residual * 255",
        early_stopping=False,
        selection="final fixed epoch; holdout evaluated once",
        augmentation=None,
        device=device,
        torch_version=torch.__version__,
        seed=args.seed,
        threads=args.threads,
    )
    return save_predictions("cnn", df, probabilities, config, time.perf_counter() - start, output)


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


def intervention_measurements(df, args, output):
    start = time.perf_counter()
    torch, device = torch_setup(args)
    model_dir = args.data_dir / "model"
    spec = importlib.util.spec_from_file_location("supplied_resnet", model_dir / "resnet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.resnet18_tin()
    model.load_state_dict(
        torch.load(model_dir / "resnet18_tin.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    model.eval().requires_grad_(False).to(device)
    classes = (model_dir / "classes.txt").read_text().splitlines()
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError("Expected 200 unique class IDs in supplied order")
    class_index = {c: i for i, c in enumerate(classes)}
    true_classes = df[df.kind == "adversarial"].set_index("id").true_class.to_dict()
    if any(c not in class_index for c in true_classes.values()):
        raise ValueError("Unknown true class in annotations")
    config = dict(
        model="supplied ResNet18TIN",
        weights_sha256=digest(model_dir / "resnet18_tin.pt"),
        implementation_sha256=digest(model_dir / "resnet.py"),
        classes_sha256=digest(model_dir / "classes.txt"),
        preprocessing="RGB [0,1]; normalization inside supplied forward only",
        output="200 logits",
        scales=list(SCALES),
        mode="eval, frozen, inference_mode",
        device=device,
        torch_version=torch.__version__,
        batch_size=32,
        threads=args.threads,
        true_class_features="oracle diagnostics only; unavailable for CLEAN/DECOY images",
    )

    def infer(images):
        result = []
        with torch.inference_mode():
            for offset in range(0, len(images), 32):
                x = torch.from_numpy(
                    np.stack(images[offset : offset + 32]).transpose(0, 3, 1, 2).astype(np.float32)
                ).to(device)
                z = model(x)
                if z.shape != (len(x), 200) or not torch.isfinite(z).all():
                    raise ValueError("Unexpected ResNet output")
                result.append(z.cpu().numpy())
        return np.concatenate(result)

    measurements = []
    groups = list(df.groupby("id", sort=False))
    for offset in range(0, len(groups), 32):
        chunk = groups[offset : offset + 32]
        originals = [read_image(args.data_dir, ident) for ident, _ in chunk]
        base = infer(originals)
        changed, metadata = [], []
        for image_num, ((ident, group), image) in enumerate(zip(chunk, originals)):
            for scale in SCALES:
                smooth = gaussian_filter(image, sigma=(scale, scale, 0), mode="reflect")
                for row in group.itertuples():
                    changed.append(
                        replace_region(image, smooth, (row.x_min, row.y_min, row.x_max, row.y_max))
                    )
                    metadata.append((image_num, row, scale))
        z = infer(changed)
        original = np.stack([base[i] for i, _, _ in metadata])
        p = torch.softmax(torch.from_numpy(original), dim=1).numpy()
        q = torch.softmax(torch.from_numpy(z), dim=1).numpy()
        pred = original.argmax(axis=1)
        modified_pred = z.argmax(axis=1)
        drop = margins(original, pred) - margins(z, pred)
        for j, (_, row, scale) in enumerate(metadata):
            y = class_index.get(true_classes.get(row.id))
            record = dict(
                id=row.id,
                region_number=int(row.region_number),
                kind=row.kind,
                scale=scale,
                partition=row.partition,
                area=int((row.x_max - row.x_min) * (row.y_max - row.y_min)),
                original_predicted_class=classes[pred[j]],
                modified_predicted_class=classes[modified_pred[j]],
                prediction_changed=bool(pred[j] != modified_pred[j]),
                predicted_margin_drop=float(drop[j]),
                predicted_probability_change=float(q[j, pred[j]] - p[j, pred[j]]),
                probability_l2_distance=float(np.linalg.norm(q[j] - p[j])),
                oracle_true_class=true_classes.get(row.id),
                oracle_original_correct=None,
                oracle_modified_correct=None,
                oracle_restored=None,
                oracle_true_margin_gain=None,
            )
            if y is not None:
                gain = float(margins(z[j : j + 1], [y])[0] - margins(original[j : j + 1], [y])[0])
                record.update(
                    oracle_original_correct=bool(pred[j] == y),
                    oracle_modified_correct=bool(modified_pred[j] == y),
                    oracle_restored=bool(pred[j] != y and modified_pred[j] == y),
                    oracle_true_margin_gain=gain,
                )
            measurements.append(record)
        if offset % 128 == 0 or offset + 32 >= len(groups):
            print(
                f"ResNet: {min(offset + 32, len(groups))}/{len(groups)} images, {time.perf_counter() - start:.1f}s",
                flush=True,
            )
            pd.DataFrame(measurements).to_csv(
                output / "resnet_interventions.partial.csv", index=False
            )
    measured = pd.DataFrame(measurements)
    feature_cols = [
        "predicted_margin_drop",
        "predicted_probability_change",
        "probability_l2_distance",
    ]
    if not np.isfinite(measured[feature_cols].to_numpy()).all():
        raise ValueError("Nonfinite intervention features")
    measured.to_csv(output / "resnet_interventions.csv", index=False)
    (output / "resnet_interventions.partial.csv").unlink(missing_ok=True)
    config["runtime_seconds"] = time.perf_counter() - start
    write_json(output / "resnet_config.json", config)
    return measured, config


def oracle_summary(measured):
    result = {}
    for partition in ("train", "validation"):
        result[partition] = {}
        for scale in SCALES:
            data = measured[(measured.partition == partition) & (measured.scale == scale)]
            known = data[data.oracle_true_margin_gain.notna()]
            summaries = {}
            for kind, group in data.groupby("kind"):
                oracle = group[group.oracle_true_margin_gain.notna()]
                wrong = oracle[oracle.oracle_original_correct.eq(False)]
                summaries[kind] = dict(
                    n=len(group),
                    known_true_class=len(oracle),
                    mean_area=float(group.area.mean()),
                    prediction_change_rate=float(group.prediction_changed.mean()),
                    mean_oracle_margin_gain=float(oracle.oracle_true_margin_gain.mean())
                    if len(oracle)
                    else None,
                    initially_wrong=len(wrong),
                    restored=int(wrong.oracle_restored.sum()),
                    restoration_rate=float(wrong.oracle_restored.mean()) if len(wrong) else None,
                )
            pairs = []
            for ident, group in known.groupby("id"):
                attack = group[group.kind == "adversarial"]
                if len(attack) != 1:
                    continue
                attack = attack.iloc[0]
                for other in group[group.kind != "adversarial"].itertuples():
                    ratio = attack.area / other.area
                    pairs.append(
                        dict(
                            id=ident,
                            comparison=other.kind,
                            area_ratio=float(ratio),
                            area_matched=bool(0.8 <= ratio <= 1.25),
                            gain_difference=float(
                                attack.oracle_true_margin_gain - other.oracle_true_margin_gain
                            ),
                            attack_larger_gain=bool(
                                attack.oracle_true_margin_gain > other.oracle_true_margin_gain
                            ),
                        )
                    )
            binary = known[known.kind != "background"]
            # Only rank known-class examples; never mix oracle scores with deployable probabilities.
            discrimination = None
            if binary.kind.nunique() == 2:
                y = binary.kind == "adversarial"
                discrimination = dict(
                    n=len(binary),
                    roc_auc=float(roc_auc_score(y, binary.oracle_true_margin_gain)),
                    average_precision=float(
                        average_precision_score(y, binary.oracle_true_margin_gain)
                    ),
                    random_average_precision=float(y.mean()),
                )
            pair_summary = {}
            for kind in ("decoy", "background"):
                for matched in (False, True):
                    selected = [
                        p
                        for p in pairs
                        if p["comparison"] == kind and (not matched or p["area_matched"])
                    ]
                    pair_summary[kind + ("_area_matched" if matched else "_all")] = dict(
                        pairs=len(selected),
                        images=len({p["id"] for p in selected}),
                        fraction_attack_larger_gain=float(
                            np.mean([p["attack_larger_gain"] for p in selected])
                        )
                        if selected
                        else None,
                        mean_gain_difference=float(
                            np.mean([p["gain_difference"] for p in selected])
                        )
                        if selected
                        else None,
                    )
            result[partition][str(scale)] = dict(
                by_kind=summaries,
                diagnostic_only_discrimination=discrimination,
                within_image=pair_summary,
            )
    return result
