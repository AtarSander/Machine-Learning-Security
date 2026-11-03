"""Noise evaluation for patch analysis experiments."""

import json
import time

import numpy as np
import pandas as pd

from src.data import COORDS as base_COORDS
from src.data import area as loc_area
from src.data import digest as base_digest
from src.data import image_proposals as loc_image_proposals
from src.data import iou as loc_iou
from src.data import read_image as base_read_image
from src.data import write_json as base_write_json
from src.noise_model import (
    GRID,
    decode,
    mask_metrics,
    max_overlap,
    predict,
    summarize_overlap,
    union_mask,
)
from src.noise_training import load_model, tensors


def phase_maps(args, config, annotations, model, device, phase):
    root = args.output_dir
    ids = config["selection"][phase]
    path = root / f"{phase}_maps.npz"
    signature = base_digest(root / "localizer.pt")
    if path.exists():
        with np.load(path) as data:
            assert data["checkpoint"].item() == signature and data["ids"].tolist() == ids
            return data["probabilities"], json.loads(data["metadata"].item())
    start = time.perf_counter()
    x, y = tensors(args, ids, annotations)
    p = predict(model, x, device, args.batch_size)
    metadata = dict(
        images=len(ids), input_and_map_seconds=time.perf_counter() - start, mask=mask_metrics(p, y)
    )
    np.savez_compressed(
        path,
        ids=np.asarray(ids),
        probabilities=p,
        checkpoint=signature,
        metadata=json.dumps(metadata),
    )
    return p, metadata


def evaluate(args, config, annotations, models, device):
    root = args.output_dir
    if (root / "localizer_metrics.json").exists():
        print("Completed proposal evaluation already exists.", flush=True)
        return
    model = load_model(root, device)
    records = []
    regions = []
    costs = {}
    masks = {}
    for phase in ("calibration", "holdout"):
        p, metadata = phase_maps(args, config, annotations, model, device, phase)
        costs[phase] = dict(**metadata)
        masks[phase] = metadata["mask"]
        decode_time = 0.0
        baseline_time = 0.0
        for n, (ident, probability) in enumerate(zip(config["selection"][phase], p)):
            start = time.perf_counter()
            boxes, scores = decode(probability, 16)
            decode_time += time.perf_counter() - start
            start = time.perf_counter()
            image = base_read_image(args.data_dir, ident)
            _, old = loc_image_proposals(image, models["fit_residual"], 16)
            baseline_time += time.perf_counter() - start
            annotation = annotations[ident]
            for method, proposals in [
                ("full_grid", list(map(tuple, GRID))),
                ("historical_residual", old),
                ("unet", boxes),
            ]:
                for index, box in enumerate(proposals):
                    records.append(
                        dict(
                            id=ident,
                            phase=phase,
                            method=method,
                            rank=index,
                            **dict(zip(base_COORDS, map(int, box))),
                            noise_score=float(scores[index]) if method == "unet" else None,
                            eval_attack_iou=loc_iou(box, annotation["box"]),
                        )
                    )
                for index, region in enumerate(annotation["regions"]):
                    regions.append(
                        dict(
                            id=ident,
                            phase=phase,
                            method=method,
                            region_number=index,
                            kind=region["kind"],
                            region_type=region["region_type"],
                            attack_family=annotation["attack_family"],
                            jpeg_quality=annotation["jpeg_quality"],
                            epsilon=region["epsilon"],
                            l2=region["l2"],
                            reported_linf=region["reported_linf"],
                            best_iou=max_overlap(proposals, region["box"]),
                        )
                    )
            if (n + 1) % 50 == 0:
                print(f"Coverage {phase}: {n + 1}/{len(p)} images", flush=True)
        costs[phase].update(
            unet_decode_seconds=decode_time,
            historical_proposals_seconds=baseline_time,
            unet_seconds_per_image=(metadata["input_and_map_seconds"] + decode_time) / len(p),
            candidates_before_suppression=len(GRID),
            candidates_after_suppression=sorted(set(len(decode(q, 16)[0]) for q in p)),
        )
    candidates = pd.DataFrame(records)
    truth = pd.DataFrame(regions)
    candidates.to_csv(root / "localizer_candidates.csv", index=False)
    truth.to_csv(root / "region_coverage.csv", index=False)
    metrics = dict(
        checkpoint_sha256=base_digest(root / "localizer.pt"),
        k=16,
        coverage={},
        subgroups={},
        costs=costs,
        mask=masks,
        prior_diagnostic_holdout_count=len(config["previously_inspected_holdout"]),
        baseline="Frozen fit_residual attack-vs-decoy classifier, historical method; never filters U-Net proposals",
        decoder_comparison="Same grid, K16, deterministic geometric suppression; no pair in grid has IoU exactly .7",
    )
    for (phase, method, kind), g in truth.groupby(["phase", "method", "kind"]):
        metrics["coverage"].setdefault(phase, {}).setdefault(method, {})[kind] = summarize_overlap(
            g.best_iou
        )
    attacked = truth[truth.kind == "adversarial"]
    for key in ("attack_family", "jpeg_quality", "epsilon", "l2", "reported_linf"):
        metrics["subgroups"][key] = {}
        for (phase, method, label), g in attacked.groupby(["phase", "method", key], dropna=False):
            metrics["subgroups"][key].setdefault(phase, {}).setdefault(method, {})[str(label)] = (
                summarize_overlap(g.best_iou)
            )
    base_write_json(root / "localizer_metrics.json", metrics)
    overlays(args, annotations, candidates)


def spatial_prior_diagnostic(args, config, annotations):
    """Post-selection diagnostic only; never changes the model, proposals, or threshold."""
    root = args.output_dir
    if (root / "spatial_prior_diagnostic.json").exists():
        return
    fit_ids = config["selection"]["fit"]
    prior = np.mean([union_mask(annotations[i])[0] for i in fit_ids], axis=0)
    np.save(root / "fitting_mean_mask.npy", prior)
    boxes, _ = decode(prior, 16)
    result = dict(
        purpose="Post-selection diagnostic of positional shortcuts; not a deployment candidate generator or tuning signal",
        source_images=len(fit_ids),
        inference_inputs="Fixed mean union mask learned only from fitting annotations",
        fixed_boxes=[list(map(int, b)) for b in boxes],
        phases={},
    )
    result["fitting_box_ranges"] = {}
    for kind in ("adversarial", "decoy"):
        fitting_boxes = [
            r["box"] for i in fit_ids for r in annotations[i]["regions"] if r["kind"] == kind
        ]
        result["fitting_box_ranges"][kind] = dict(
            count=len(fitting_boxes),
            minimum_y_min=min(b[1] for b in fitting_boxes),
            maximum_y_max=max(b[3] for b in fitting_boxes),
        )
    for phase in ("calibration", "holdout"):
        ids = config["selection"][phase]
        if not (root / f"{phase}_maps.npz").exists():
            continue
        attack = []
        decoy = []
        for ident in ids:
            a = annotations[ident]
            if loc_area(a["box"]):
                attack.append(max_overlap(boxes, a["box"]))
            decoy.extend(max_overlap(boxes, r["box"]) for r in a["regions"] if r["kind"] == "decoy")
        with np.load(root / f"{phase}_maps.npz") as data:
            maps = data["probabilities"]
        mean_map = maps.mean(0)
        result["phases"][phase] = dict(
            attack=summarize_overlap(attack),
            decoy=summarize_overlap(decoy),
            mean_pixel_std_across_images=float(maps.std(0).mean()),
            mean_absolute_deviation_from_average_map=float(np.abs(maps - mean_map).mean()),
            correlation_mean_prediction_with_fit_prior=float(
                np.corrcoef(prior.ravel(), mean_map.ravel())[0, 1]
            ),
        )
    base_write_json(root / "spatial_prior_diagnostic.json", result)
    from src.data import plt

    with np.load(root / "calibration_maps.npz") as data:
        maps = data["probabilities"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, values, title in zip(
        axes,
        [prior, maps.mean(0), maps.std(0)],
        ["Fitting mean union mask", "Mean calibration U-Net map", "U-Net variation across images"],
    ):
        im = ax.imshow(values, vmin=0, vmax=1 if "variation" not in title else None, cmap="magma")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(root / "spatial_prior_diagnostic.png", dpi=150)
    plt.close(fig)


def report(args, config):
    root = args.output_dir
    lines = [
        "# Trained union-noise U-Net experiment",
        "",
        "The U-Net predicts attack + decoy union masks from RGB and signed full-image residuals. The attack classifier remains frozen. "
        "No IoU regressor, synthetic attacks, or test data are used.",
        "",
        f"Full-partition experiment: {config['full_experiment']}. Actual image counts: "
        + str({k: len(v) for k, v in config["selection"].items()})
        + ".",
        "Persisted split: 3200 fitting / 800 calibration / 1000 holdout. The earlier 40 holdout images were already inspected diagnostically; no assignments changed.",
        "RGB scale=1, residual scale=10 fixed in advance; Gaussian sigmas 0.7 and 1.2, reflect boundaries. No augmentation.",
        "Checkpoint selection uses calibration attack recall at IoU >=0.5, K16, with mean best attack IoU as tie-breaker.",
        "The recovered reference scorer saw all original training images and is diagnostic only. The main pipeline uses the unchanged fit_combined scorer trained on the 3200 fitting images.",
        "",
    ]
    if (root / "training.json").exists():
        t = json.loads((root / "training.json").read_text())
        lines += [
            f"Training: {t['device']}, {t['parameters']} parameters, {t['epochs_completed']} epochs, selected epoch {t['selected_epoch']}; {t['runtime_seconds']:.1f} seconds.",
            "![Learning curves](learning_curve.png)",
            "",
        ]
    if (root / "localizer_metrics.json").exists():
        m = json.loads((root / "localizer_metrics.json").read_text())
        lines += [
            "## Proposal coverage before the attack classifier",
            "",
            "| Phase | Proposal method | Target | Count | Mean best IoU | Recall .3 | Recall .5 | Recall .7 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for phase, methods in m["coverage"].items():
            for method, kinds in methods.items():
                for kind, r in kinds.items():
                    lines.append(
                        f"| {phase} | {method} | {kind} | {r['count']} | {r['mean_best_iou']:.4f} | {r['recall']['0.3']:.4f} | {r['recall']['0.5']:.4f} | {r['recall']['0.7']:.4f} |"
                    )
        lines += [
            "",
            "Attack counts are images; decoy counts are individual annotated rectangles. All methods use identical IDs and the same 265-box grid; shortlists use K16. Historical residual attack-vs-decoy scores never filter U-Net proposals.",
            "",
            "| Phase | Union-mask IoU | Nonempty-mask IoU | Map + decode seconds/image |",
            "|---|---:|---:|---:|",
        ]
        for phase, c in m["costs"].items():
            lines.append(
                f"| {phase} | {m['mask'][phase]['mean_mask_iou']:.4f} | {m['mask'][phase]['nonempty_mask_iou']:.4f} | {c['unet_seconds_per_image']:.5f} |"
            )
        lines += [
            "",
            "![Proposal failures and no-attack maps](localizer_overlays.png)",
            "",
            "Attack-family, strength, and JPEG coverage strata with counts are saved in localizer_metrics.json. Maps and individual decoy coverage are saved separately.",
        ]
        lines += [
            "",
            "| Holdout attack family | Attacks | U-Net best IoU | U-Net recall .5 | Historical recall .5 |",
            "|---|---:|---:|---:|---:|",
        ]
        families = m["subgroups"]["attack_family"]["holdout"]
        for name, r in families["unet"].items():
            old = families["historical_residual"][name]
            lines.append(
                f"| {name} | {r['count']} | {r['mean_best_iou']:.4f} | {r['recall']['0.5']:.4f} | {old['recall']['0.5']:.4f} |"
            )
    if (root / "pipeline_metrics.json").exists():
        p = json.loads((root / "pipeline_metrics.json").read_text())
        h = p["holdout"]
        lines += [
            "",
            "## Frozen attack-classifier pipeline",
            "",
            f"Calibration-selected threshold: {p['selected_threshold']['threshold']:.8f}. Calibration official score: {p['selected_threshold']['official_mean']:.4f}.",
            f"Holdout official score: {h['official']['official_mean']:.4f}; all-zero baseline: {h['official']['all_zero_baseline']:.4f}.",
            f"Mean best available attack IoU: {h['candidates']['mean_best_iou']:.4f}; selected IoU before rejection: {h['candidates']['mean_selected_iou']:.4f}; selection gap: {h['candidates']['selection_gap']:.4f}.",
            "",
            "| Group | Images | Mean official score | Abstention rate | Image positive rate |",
            "|---|---:|---:|---:|---:|",
        ]
        for kind, g in h["official"]["groups"].items():
            lines.append(
                f"| {kind} | {g['images']} | {g['official_mean']:.4f} | {g['abstention_rate']:.4f} | {g['any_candidate_exceeds_threshold']:.4f} |"
            )
        lines += [
            "",
            "The image positive rate is a false-positive rate for CLEAN and DECOY only. Official score is attack IoU (zero on rejection), or one for a zero box on no-attack images.",
            "Classifier timing and forward counts are in pipeline_metrics.json; resumed runs can reuse candidate features.",
        ]
        lines += [
            "",
            "| Pipeline phase | Classifier seconds | Original forward images | Modified forward images | Cached candidates |",
            "|---|---:|---:|---:|---:|",
        ]
        for phase, t in p["timing"].items():
            c = t["forward_counts"]
            lines.append(
                f"| {phase} | {t['seconds']:.1f} | {c['original_forward_images']} | {c['intervention_forward_images']} | {c['cached_candidates']} |"
            )
        lines += [
            "",
            "## Interpretation",
            "",
            "Proposal coverage and final classification are different measurements. The U-Net is supervised with the union of all annotated noise; "
            "the frozen second stage must still distinguish attacks from decoys and choose accurate boundaries. "
            "Its confidence is not a prediction of box IoU. The full grid sets the geometric ceiling for this decoder.",
        ]
        if (root / "localizer_metrics.json").exists():
            coverage_metrics = json.loads((root / "localizer_metrics.json").read_text())[
                "coverage"
            ]["holdout"]
            new = coverage_metrics["unet"]["adversarial"]
            old = coverage_metrics["historical_residual"]["adversarial"]
            lines += [
                f"On identical holdout images, U-Net changes attack recall at IoU .5 by "
                f"{100 * (new['recall']['0.5'] - old['recall']['0.5']):+.2f} percentage points relative to the historical shortlist. "
                f"The remaining selection gap among U-Net candidates is {h['candidates']['selection_gap']:.4f} mean IoU. "
                "These observations justify separating future proposal improvements from future classifier/boundary-ranking changes; "
                "no changes were selected using this holdout."
            ]
    else:
        lines += [
            "",
            "Full frozen-classifier pipeline evaluation has not completed in this output directory. No final image score is reported yet.",
        ]
    if (root / "spatial_prior_diagnostic.json").exists():
        d = json.loads((root / "spatial_prior_diagnostic.json").read_text())
        lines += [
            "",
            "## Positional-shortcut diagnostic",
            "",
            "Failure overlays showed similar maps even on clean images. After checkpoint selection, a fixed mean-mask diagnostic was constructed from fitting annotations only. "
            "It uses the same grid/decoder/K16 but no image pixels. This diagnostic did not change the chosen model, deployment candidates, or threshold.",
            "",
            "| Phase | Fixed fitting-prior attack recall .5 | Mean best attack IoU | Mean U-Net pixel standard deviation across images |",
            "|---|---:|---:|---:|",
        ]
        for phase, r in d["phases"].items():
            lines.append(
                f"| {phase} | {r['attack']['recall']['0.5']:.4f} | {r['attack']['mean_best_iou']:.4f} | {r['mean_pixel_std_across_images']:.5f} |"
            )
        lines += [
            "",
            "![Positional diagnostic](spatial_prior_diagnostic.png)",
            "",
            "Coverage gains over the historical attack-versus-decoy shortlist alone do not establish image-dependent noise recognition. "
            "A competitive fixed spatial prior and low map variation indicate that positional regularities explain much of the gain. "
            "The selected checkpoint remains a trained U-Net, but its content sensitivity is a material limitation; "
            "future training work should include this spatial baseline in calibration diagnostics.",
        ]
        if "fitting_box_ranges" in d:
            lines += [
                "",
                f"Observed fitting-annotation ranges: {d['fitting_box_ranges']}. This is a property of the fitting data, not an assumption about test images.",
            ]
    lines += [
        "",
        "K32/K64 comparisons, connected-component boxes, boundary refinement, classifier adaptation, and IoU regression were not run. "
        "The checkpoint and threshold use calibration data only; the final holdout result is not used to tune settings. "
        "A small smoke run is an implementation check, not the full experiment. Earlier supplied known-box measurements remain historical references, not newly reproduced results.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def overlays(args, annotations, candidates):
    from src.data import Rectangle, plt

    root = args.output_dir
    with np.load(root / "holdout_maps.npz") as data:
        maps = dict(zip(data["ids"].tolist(), data["probabilities"]))
    unet = candidates[(candidates.phase == "holdout") & (candidates.method == "unet")]
    attacked = [i for i in maps if loc_area(annotations[i]["box"])]
    best = unet.groupby("id").eval_attack_iou.max()
    ids = sorted(attacked, key=lambda i: (best[i], i))[:4]
    ids += [
        next(i for i in maps if annotations[i]["image_kind"] == kind) for kind in ("DECOY", "CLEAN")
    ]
    fig, axes = plt.subplots(
        len(ids), 3, figsize=(10, 3 * len(ids)), layout="constrained", squeeze=False
    )
    for n, ident in enumerate(ids):
        image = base_read_image(args.data_dir, ident)
        annotation = annotations[ident]
        axes[n, 0].imshow(image)
        axes[n, 1].imshow(maps[ident], vmin=0, vmax=1, cmap="magma")
        axes[n, 2].imshow(image)
        for ax in axes[n]:
            for r in annotation["regions"]:
                x, y, xx, yy = r["box"]
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        xx - x,
                        yy - y,
                        fill=False,
                        edgecolor="red" if r["kind"] == "adversarial" else "cyan",
                        linewidth=1.5,
                    )
                )
            ax.axis("off")
        for rank, (_, r) in enumerate(unet[unet.id == ident].iterrows()):
            axes[n, 2].add_patch(
                Rectangle(
                    (r.x_min - 0.5, r.y_min - 0.5),
                    r.x_max - r.x_min,
                    r.y_max - r.y_min,
                    fill=False,
                    edgecolor="gold",
                    alpha=0.5,
                    linewidth=0.6,
                )
            )
        axes[n, 0].set_title(f"{ident}: {annotation['image_kind']}")
        axes[n, 1].set_title("U-Net noise probability")
        axes[n, 2].set_title(f"16 proposals; best attack IoU {best[ident]:.3f}")
    fig.suptitle(
        "Worst attacked coverage plus no-attack examples. True attacks red; decoys cyan; proposals gold."
    )
    fig.savefig(root / "localizer_overlays.png", dpi=140)
    plt.close(fig)
