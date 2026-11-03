"""Noise training for patch analysis experiments."""

import csv
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from src.candidates import selected_images as loc_selected_images
from src.data import COORDS as base_COORDS
from src.data import LABEL_FREE as loc_LABEL_FREE
from src.data import digest as base_digest
from src.data import parse_comment as base_parse_comment
from src.data import read_image as base_read_image
from src.data import residual_columns as base_residual_columns
from src.data import write_json as base_write_json
from src.noise_model import (
    DECODER,
    PREPROCESS,
    TRANSFORMS,
    UNet,
    channels,
    coverage,
    loss_parts,
    mask_metrics,
    predict,
    transform_annotation,
    transform_tensor,
    union_mask,
)


def annotations_at(data_dir):
    result = {}
    with (data_dir / "train_labels.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            box = tuple(int(row[k]) for k in base_COORDS)
            regions, meta = base_parse_comment(row["comment"], box)
            if row["id"] in result:
                raise ValueError("Duplicate image id")
            result[row["id"]] = dict(box=box, regions=regions, **meta)
    return result


def prepare(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    source = args.localizer_source
    manifest = json.loads((source / "split_manifest.json").read_text())
    ids = [
        set(manifest[k]) for k in ("fit_image_ids", "calibration_image_ids", "holdout_image_ids")
    ]
    assert [len(s) for s in ids] == [3200, 800, 1000]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert ids[2] == set(manifest["original_split"]["validation_image_ids"])
    annotations = annotations_at(args.data_dir)
    assert set.union(*ids) == set(annotations)
    selection = loc_selected_images(manifest, annotations, args.limit, args.seed)
    audit = json.loads((source / "reference/audit.json").read_text())
    checkpoint = source / "reference/scorers.joblib"
    assert base_digest(checkpoint) == audit["checkpoint_sha256"]
    columns = (
        base_residual_columns(pd.DataFrame(columns=audit["explicit_features"])) + loc_LABEL_FREE
    )
    assert columns == audit["explicit_features"]
    augmented = getattr(args, "augment", False)
    config = dict(
        seed=args.seed,
        selection=selection,
        preprocessing=PREPROCESS,
        decoder=DECODER,
        architecture=dict(
            widths=[16, 32, 64, 128],
            input_channels=9,
            output_channels=1,
            upsampling="bilinear",
            normalization="none",
        ),
        optimizer=dict(name="AdamW", lr=0.001, weight_decay=0.0001),
        batch_size=args.batch_size,
        maximum_epochs=args.localizer_epochs,
        patience=5,
        loss="BCEWithLogits + soft Dice, equal weights, smoothing=1",
        checkpoint_selection="Calibration attack recall IoU>=0.5 at K16; tie-break mean best attack IoU"
        + ("; both averaged over all eight dihedral orientations" if augmented else ""),
        labels_sha256=base_digest(args.data_dir / "train_labels.csv"),
        split_sha256=base_digest(source / "split_manifest.json"),
        attack_scorers_sha256=audit["checkpoint_sha256"],
        features=columns,
        source=str(source.resolve()),
        full_experiment=args.limit is None,
        previously_inspected_holdout=json.loads((source / "selection.json").read_text())[
            "selection"
        ]["holdout"],
    )
    if augmented:
        config["augmentation"] = (
            "Per-sample random dihedral transform (flips, transposes, 90-degree rotations) of input channels and masks during fitting only"
        )
    # JSON normalizes tuples in the decoder for repeatable comparisons.
    config = json.loads(json.dumps(config))
    if (root / "config.json").exists() and json.loads((root / "config.json").read_text()) != config:
        raise ValueError(
            "Output belongs to another localizer configuration; use a new output directory"
        )
    base_write_json(root / "config.json", config)
    base_write_json(root / "split_manifest.json", manifest)
    return config, annotations, joblib.load(checkpoint)


def tensors(args, ids, annotations):
    x = np.empty((len(ids), 9, 64, 64), np.float32)
    y = np.empty((len(ids), 1, 64, 64), np.float32)
    for n, ident in enumerate(ids):
        x[n] = channels(base_read_image(args.data_dir, ident))
        y[n] = union_mask(annotations[ident])
    return torch.from_numpy(x), torch.from_numpy(y)


def augment(bx, by, generator):
    """Apply an independently drawn dihedral transform to every sample (tensors only; nothing is recomputed)."""
    ks = torch.randint(len(TRANSFORMS), (len(bx),), generator=generator)
    bx, by = bx.clone(), by.clone()
    for k in TRANSFORMS[1:]:
        chosen = ks == k
        if chosen.any():
            bx[chosen], by[chosen] = (
                transform_tensor(bx[chosen], k),
                transform_tensor(by[chosen], k),
            )
    return bx, by


def dihedral_coverage(model, cx, ids, annotations, device, batch_size):
    """Calibration attack coverage in all eight orientations, so selection cannot reward a positional prior."""
    per = []
    for k in TRANSFORMS:
        moved = {i: transform_annotation(annotations[i], k) for i in ids}
        p = predict(model, transform_tensor(cx, k).contiguous(), device, batch_size)
        per.append(coverage(p, ids, moved)["attack"])
    return dict(
        per_transform=per,
        recall=float(np.mean([a["recall"]["0.5"] for a in per])),
        mean_best_iou=float(np.mean([a["mean_best_iou"] for a in per])),
    )


def save_torch(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args, config, annotations, device):
    root = args.output_dir
    if (root / "training.json").exists():
        print("Completed training already exists; reusing selected checkpoint.", flush=True)
        return
    if (root / "last.pt").exists() and not args.resume:
        raise ValueError("Training state exists; use --resume to continue")
    start = time.perf_counter()
    print("Loading complete-image fitting and calibration tensors...", flush=True)
    fit_ids, cal_ids = config["selection"]["fit"], config["selection"]["calibration"]
    x, y = tensors(args, fit_ids, annotations)
    cx, cy = tensors(args, cal_ids, annotations)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    best = (-1.0, -1.0)
    stale = 0
    first = 1
    prior_seconds = 0.0
    if args.resume and (root / "last.pt").exists():
        state = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        generator.set_state(state["generator"])
        history = state["history"]
        best = tuple(state["best"])
        stale = state["stale"]
        first = state["epoch"] + 1
        prior_seconds = state["elapsed_seconds"]
    augmented = getattr(args, "augment", False)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for epoch in range(first, args.localizer_epochs + 1):
        if stale >= 5:
            break
        tick = time.perf_counter()
        model.train()
        total = 0.0
        for batch, (bx, by) in enumerate(loader):
            if augmented:
                bx, by = augment(bx, by, generator)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_parts(model(bx.to(device)), by.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(bx)
            if (batch + 1) % 25 == 0:
                print(
                    f"Epoch {epoch}: batch {batch + 1}/{len(loader)}, mean loss {total / min((batch + 1) * args.batch_size, len(x)):.4f}",
                    flush=True,
                )
        probabilities = predict(model, cx, device, args.batch_size)
        metrics = coverage(probabilities, cal_ids, annotations)
        metric = (metrics["attack"]["recall"]["0.5"], metrics["attack"]["mean_best_iou"])
        dihedral = None
        if augmented:
            dihedral = dihedral_coverage(model, cx, cal_ids, annotations, device, args.batch_size)
            metric = (dihedral["recall"], dihedral["mean_best_iou"])
        selected = metric > best
        record = dict(
            epoch=epoch,
            training_loss=total / len(x),
            calibration_mask=mask_metrics(probabilities, cy),
            calibration_coverage=metrics,
            calibration_dihedral=dihedral,
            selected=selected,
            seconds=time.perf_counter() - tick,
        )
        history.append(record)
        if selected:
            best = metric
            stale = 0
            save_torch(
                root / "localizer.pt",
                dict(model=model.state_dict(), epoch=epoch, config=config, calibration=record),
            )
        else:
            stale += 1
        save_torch(
            root / "last.pt",
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                generator=generator.get_state(),
                epoch=epoch,
                best=best,
                stale=stale,
                history=history,
                elapsed_seconds=prior_seconds + time.perf_counter() - start,
            ),
        )
        base_write_json(root / "learning_curve.json", history)
        print(
            f"Epoch {epoch}: train={record['training_loss']:.4f}, calibration attack recall@16/0.5={metric[0]:.4f}, best IoU={metric[1]:.4f}, mask IoU={record['calibration_mask']['mean_mask_iou']:.4f}, {record['seconds']:.1f}s, selected={selected}, patience={stale}/5",
            flush=True,
        )
    best_state = torch.load(root / "localizer.pt", map_location="cpu", weights_only=False)
    base_write_json(
        root / "training.json",
        dict(
            fit_images=len(x),
            calibration_images=len(cx),
            device=device,
            parameters=sum(p.numel() for p in model.parameters()),
            epochs_completed=len(history),
            selected_epoch=best_state["epoch"],
            best_calibration=best_state["calibration"],
            runtime_seconds=prior_seconds + time.perf_counter() - start,
            checkpoint_sha256=base_digest(root / "localizer.pt"),
            early_stopped=stale >= 5,
            history=history,
        ),
    )
    plot_learning(root, history)


def load_model(root, device):
    training = json.loads((root / "training.json").read_text())
    if base_digest(root / "localizer.pt") != training["checkpoint_sha256"]:
        raise ValueError("Selected localizer checkpoint changed")
    state = torch.load(root / "localizer.pt", map_location="cpu", weights_only=False)
    if state["config"]["preprocessing"] != PREPROCESS:
        raise ValueError("Preprocessing changed")
    model = UNet().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def plot_learning(root, history):
    from src.data import plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["training_loss"] for r in history], label="fit")
    axes[0].plot(
        epochs, [r["calibration_mask"]["total_loss"] for r in history], label="calibration"
    )
    axes[0].legend()
    axes[0].set_title("BCE + Dice")
    axes[1].plot(epochs, [r["calibration_coverage"]["attack"]["recall"]["0.5"] for r in history])
    axes[1].set_title("Calibration attack recall @16 / IoU .5")
    axes[2].plot(epochs, [r["calibration_mask"]["mean_mask_iou"] for r in history])
    axes[2].set_title("Calibration union-mask IoU")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.savefig(root / "learning_curve.png", dpi=150)
    plt.close(fig)
