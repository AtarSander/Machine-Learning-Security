"""Localization for patch analysis experiments."""

import json
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor

from src.candidates import (
    CandidateScorer,
    add_scores,
    annotate,
    fit_logistic,
    prepare_reference,
    selected_images,
)
from src.checks import localization_checks as self_test
from src.data import COORDS as base_COORDS
from src.data import (
    LABEL_FREE,
    SIZES,
    area,
    background_boxes,
    grid_boxes,
    image_proposals,
    output_box,
    stable_rng,
    variants,
)
from src.data import SCALES as base_SCALES
from src.data import digest as base_digest
from src.data import read_image as base_read_image
from src.data import residual_columns as base_residual_columns
from src.data import residual_maps as base_residual_maps
from src.data import write_json as base_write_json
from src.metrics import (
    calibrate,
    grid_coverage,
    original_benchmark,
    phase_reports,
    pipeline_metrics,
    pool_metrics,
    selected_rows,
    subgroup_metrics,
)
from src.metrics import experiment_metrics as base_experiment_metrics
from src.reports import write_report


def adapt(rows, extra, columns, manifest, args):
    known = rows[(rows.kind != "background") & rows.id.isin(manifest["fit_image_ids"])]
    extra = extra[(extra.phase == "fit") & ((extra.eval_iou >= 0.5) | (extra.eval_iou <= 0.1))]
    assert set(extra.id) <= set(manifest["fit_image_ids"])
    data = pd.concat([known[columns], extra[columns]], ignore_index=True)
    target = np.concatenate(
        [(known.kind == "adversarial").to_numpy(), (extra.eval_iou >= 0.5).to_numpy()]
    )
    config = dict(
        known_attack_regions=int((known.kind == "adversarial").sum()),
        known_decoy_negatives=int((known.kind == "decoy").sum()),
        added_positive_regions=int((extra.eval_iou >= 0.5).sum()),
        added_negative_regions=int((extra.eval_iou <= 0.1).sum()),
        added_images=int(extra.id.nunique()),
        columns=columns,
        C=1.0,
        max_iter=2000,
        seed=args.seed,
    )
    return fit_logistic(data, columns, target, args.seed), config


def robustness(args, root, rows, columns, models, manifest, annotations, selection, engine):
    records = []
    for phase, ids in selection.items():
        for j, ident in enumerate(ids):
            annotation = annotations[ident]
            if not annotation["regions"]:
                continue
            exact = {tuple(r["box"]): r["kind"] for r in annotation["regions"]}
            sources = {}
            for n, r in enumerate(annotation["regions"]):
                for b in variants(
                    r["box"], args.candidate_budget, stable_rng(args.seed, ident + str(n))
                ):
                    if b not in exact:
                        sources.setdefault(b, set()).add(r["kind"] + ":" + r["region_type"])
            boxes = sorted(set(exact) | set(sources))
            table = engine.extract(base_read_image(args.data_dir, ident), boxes)
            table["is_exact"] = [b in exact for b in boxes]
            table["candidate_origin"] = [
                "exact_" + exact[b]
                if b in exact
                else "variants_from_" + ";".join(sorted(sources[b]))
                for b in boxes
            ]
            records.append(annotate(table, ident, phase, annotation))
            if (j + 1) % 10 == 0:
                print(f"Robustness {phase}: {j + 1}/{len(ids)} images", flush=True)
    table = add_scores(pd.concat(records, ignore_index=True), models, columns)
    calibrated = table[(table.phase == "calibration") & ~table.is_exact]
    check = pool_metrics(calibrated, "fit_combined")
    adaptation = dict(run=False, reason="No calibration selection gap")
    if check.get("selection_gap") is not None and check["selection_gap"] > 0:
        model, config = adapt(rows, table[~table.is_exact], columns, manifest, args)
        models = {**models, "adapted_robustness": model}
        table["adapted_robustness"] = model.predict_proba(table[columns])[:, 1]
        joblib.dump(model, root / "robustness_model.joblib")
        adaptation = dict(
            run=True,
            reason="Reserved calibration selected IoU below achievable best IoU",
            configuration=config,
        )
    result = dict(
        annotation_seeded=True,
        budget_per_source_region=args.candidate_budget,
        candidate_targets="IoU>=0.5 positive; IoU<=0.1 negative; middle excluded only for binary metrics",
        adaptation=adaptation,
        pools={},
        original_benchmark=original_benchmark(rows, selection, models, columns),
    )
    for name in ("frozen_reference", "fit_combined", "adapted_robustness"):
        if name not in table:
            continue
        result["pools"][name] = {
            pool: phase_reports(part, name)
            for pool, part in [
                ("exact_only", table[table.is_exact]),
                ("perturbed_only", table[~table.is_exact]),
                ("exact_and_perturbed", table),
            ]
        }
    result["holdout_subgroups"] = subgroup_metrics(
        table[(table.phase == "holdout") & ~table.is_exact], "frozen_reference"
    )
    table.to_csv(root / "robustness_candidates.csv", index=False)
    return result


def background(args, root, rows, columns, models, manifest, annotations, selection, engine):
    records = []
    for phase, ids in selection.items():
        for j, ident in enumerate(ids):
            annotation = annotations[ident]
            image = base_read_image(args.data_dir, ident)
            marked = {tuple(r["box"]): "exact_" + r["kind"] for r in annotation["regions"]}
            controls = background_boxes(
                image,
                list(marked),
                args.candidate_budget,
                stable_rng(args.seed, ident + "background"),
            )
            ordered = list(marked) + [b for b, _ in controls]
            origins = list(marked.values()) + [origin for _, origin in controls]
            assert len(ordered) == len(set(ordered))
            table = engine.extract(image, ordered)
            table["candidate_origin"] = origins
            table["order_in_pool"] = np.arange(len(table))
            table["control_order"] = [-1] * len(marked) + list(range(len(controls)))
            records.append(annotate(table, ident, phase, annotation))
            if (j + 1) % 10 == 0:
                print(f"Background {phase}: {j + 1}/{len(ids)} images", flush=True)
    table = add_scores(pd.concat(records, ignore_index=True), models, columns)
    controls = table[table.control_order >= 0]
    # Train a single fixed background adaptation; original decoys remain the main negative population.
    model, config = adapt(rows, controls, columns, manifest, args)
    models = {**models, "adapted_background": model}
    table["adapted_background"] = model.predict_proba(table[columns])[:, 1]
    joblib.dump(model, root / "background_model.joblib")
    result = dict(
        annotation_seeded=True,
        adaptation=config,
        budgets={},
        background_types={},
        original_benchmark=original_benchmark(rows, selection, models, columns),
    )
    for origin, g in table[table.control_order >= 0].groupby("candidate_origin"):
        result["background_types"][origin] = {
            name: phase_reports(g, name)
            for name in ("frozen_reference", "fit_combined", "adapted_background")
        }
    for budget in sorted({b for b in (4, 8, args.candidate_budget) if b <= args.candidate_budget}):
        mixed = table[table.order_in_pool < budget].copy()
        control_pool = table[(table.control_order >= 0) & (table.control_order < budget)]
        detail = dict(
            actual_candidates_by_image=mixed.groupby("id").size().to_dict(),
            control_only=phase_reports(control_pool, "frozen_reference"),
            frozen_at_half=phase_reports(mixed, "frozen_reference"),
            calibrated={},
        )
        for name in ("fit_combined", "adapted_background"):
            best, curve = calibrate(mixed[mixed.phase == "calibration"], name)
            final, selected = pipeline_metrics(
                mixed[mixed.phase == "holdout"], name, best["threshold"]
            )
            negatives = mixed[(mixed.phase == "holdout") & (mixed.eval_iou <= 0.1)]
            final["candidate_negative_fpr"] = (
                float((negatives[name] >= best["threshold"]).mean()) if len(negatives) else None
            )
            detail["calibrated"][name] = dict(
                calibration=best, threshold_curve=curve, holdout=final
            )
            selected.to_csv(root / f"background_{budget}_{name}_holdout.csv", index=False)
        result["budgets"][str(budget)] = detail
    comparable = table[
        (table.image_kind == "ADV")
        & ((table.control_order >= 0) | (table.candidate_origin == "exact_adversarial"))
    ]
    result["attack_versus_background"] = {
        name: phase_reports(comparable, name)
        for name in ("frozen_reference", "fit_combined", "adapted_background")
    }
    table.to_csv(root / "background_candidates.csv", index=False)
    return result


def ablation(args, root, rows, columns, models, manifest, annotations, selection, engine):
    heldout = rows[(rows.kind != "background") & rows.id.isin(selection["holdout"])].copy()
    metrics = {}
    for name, features in [
        ("residual_reference", base_residual_columns(rows)),
        ("resnet_reference", LABEL_FREE),
        ("frozen_reference", columns),
    ]:
        start = time.perf_counter()
        heldout["probability"] = models[name].predict_proba(heldout[features])[:, 1]
        m = base_experiment_metrics(heldout, "probability")
        m.update(
            features=features,
            prediction_seconds=time.perf_counter() - start,
            feature_cost="No network inference"
            if name == "residual_reference"
            else "One cached original prediction per image, two full-image interventions per region",
            measured_forward_images=0,
            execution="Replay of archived features; no fresh neural feature computation",
            training="All original training images; identical annotated attack/decoy samples and LogisticRegression C=1, max_iter=2000",
        )
        metrics[name] = m
        heldout[["id", "region_number", "kind", "partition", "probability"]].to_csv(
            root / f"ablation_{name}_predictions.csv", index=False
        )
    metrics["combined_minus_resnet_only"] = {
        key: (
            metrics["frozen_reference"]["overall"][key]
            - metrics["resnet_reference"]["overall"][key]
            if metrics["frozen_reference"]["overall"][key] is not None
            and metrics["resnet_reference"]["overall"][key] is not None
            else None
        )
        for key in ("roc_auc", "average_precision", "strict_ranking_accuracy")
    }
    return metrics


def proposals(args, root, rows, columns, models, manifest, annotations, selection, engine):
    records, grid_records = [], []
    for phase, ids in selection.items():
        for j, ident in enumerate(ids):
            image = base_read_image(args.data_dir, ident)
            grid, boxes = image_proposals(image, models["fit_residual"], args.candidate_budget)
            # Annotations first enter after generation/shortlisting, solely for evaluation.
            table = annotate(engine.extract(image, boxes), ident, phase, annotations[ident])
            records.append(table)
            full = annotate(
                pd.DataFrame(grid, columns=base_COORDS), ident, phase, annotations[ident]
            )
            full["shortlisted"] = [b in boxes for b in grid]
            grid_records.append(full)
            if (j + 1) % 10 == 0:
                print(f"Proposals {phase}: {j + 1}/{len(ids)} images", flush=True)
    table = add_scores(pd.concat(records, ignore_index=True), models, columns)
    full = pd.concat(grid_records, ignore_index=True)
    calibration = selected_rows(
        table[(table.phase == "calibration") & table.eval_has_attack], "fit_combined"
    )
    ratios = (
        (calibration.x_max - calibration.x_min) * (calibration.y_max - calibration.y_min)
    ) / calibration.eval_true_area
    median_ratio = float(ratios.median()) if len(ratios) else None
    regression = dict(
        run=False,
        calibration_median_selected_area_ratio=median_ratio,
        trigger="Median selected/true area on calibration attacked images exceeds 1.1",
        refinement="not run",
    )
    eligible = ["fit_combined"]
    if median_ratio is not None and median_ratio > 1.1:
        train = table[table.phase == "fit"]
        assert set(train.id) <= set(manifest["fit_image_ids"])
        cfg = dict(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=args.seed,
        )
        regressor = HistGradientBoostingRegressor(**cfg)
        regressor.fit(train[columns], train.eval_iou)
        table["quality_regressor"] = np.clip(regressor.predict(table[columns]), 0, 1)
        joblib.dump(regressor, root / "quality_regressor.joblib")
        eligible.append("quality_regressor")
        regression.update(
            run=True,
            configuration=cfg,
            training_images=int(train.id.nunique()),
            training_candidates=len(train),
            features=columns,
        )
    candidate_results = {
        name: phase_reports(table, name) for name in ["frozen_reference"] + eligible
    }
    calibrated = {}
    for name in eligible:
        best, curve = calibrate(table[table.phase == "calibration"], name)
        final, selected = pipeline_metrics(table[table.phase == "holdout"], name, best["threshold"])
        selected.to_csv(root / f"proposals_{name}_holdout.csv", index=False)
        calibrated[name] = dict(calibration=best, threshold_curve=curve, holdout=final)
    winner = max(eligible, key=lambda name: calibrated[name]["calibration"]["official_mean"])
    threshold = calibrated[winner]["calibration"]["threshold"]
    summary, predictions = pipeline_metrics(table[table.phase == "holdout"], winner, threshold)
    predictions[["id"] + ["output_" + c for c in base_COORDS] + [winner]].to_csv(
        root / "holdout_boxes.csv", index=False
    )
    pipeline = dict(
        scorer=winner,
        threshold=threshold,
        candidate_budget=args.candidate_budget,
        size_bank=SIZES,
        stride=8,
        shortlisting="Training-only residual logistic score, greedy IoU<0.7 diversity, deterministic fill",
        features=columns,
        checkpoint_sha256=base_digest(root / "reference/scorers.joblib"),
        selection="Maximum calibration official mean; ties retain combined scorer; threshold ties prefer rejection",
        inference="Image pixels only; no annotations, true class, source labels or IoUs",
    )
    if winner == "quality_regressor":
        pipeline["quality_checkpoint_sha256"] = base_digest(root / "quality_regressor.joblib")
    base_write_json(root / "pipeline.json", pipeline)
    table.to_csv(root / "proposal_candidates.csv", index=False)
    full.to_csv(root / "proposal_grid.csv", index=False)
    result = dict(
        annotation_seeded=False,
        grid_candidates_per_image=len(grid_boxes()),
        scored_budget=args.candidate_budget,
        size_bank=SIZES,
        stride=8,
        before_shortlisting=grid_coverage(full),
        after_shortlisting=grid_coverage(full[full.shortlisted]),
        candidate_models=candidate_results,
        quality_regression=regression,
        calibrated=calibrated,
        selected_pipeline=pipeline,
        final_holdout=summary,
        holdout_subgroups=subgroup_metrics(table[table.phase == "holdout"], winner),
        not_run=[
            "Local boundary refinement",
            "Learned localization/segmentation network",
            "Attack generation",
            "Test-image inference or submissions",
        ],
    )
    failure_gallery(predictions, annotations, args, root)
    return result


def infer_image(args):
    root = args.output_dir
    config = json.loads((root / "pipeline.json").read_text())
    if base_digest(root / "reference/scorers.joblib") != config["checkpoint_sha256"]:
        raise ValueError("Inference checkpoint changed since calibration")
    if config["size_bank"] != [list(s) for s in SIZES] or config["stride"] != 8:
        raise ValueError("Proposal generator changed since calibration")
    models = joblib.load(root / "reference/scorers.joblib")
    with Image.open(args.image) as im:
        if im.mode != "RGB" or im.size != (64, 64):
            raise ValueError("Expected a native 64x64 RGB image")
        image = np.asarray(im, dtype=np.float64) / 255
    _, boxes = image_proposals(image, models["fit_residual"], config["candidate_budget"])
    engine = CandidateScorer(args, root / "cache")
    features = engine.extract(image, boxes)
    if config["scorer"] == "quality_regressor":
        if base_digest(root / "quality_regressor.joblib") != config["quality_checkpoint_sha256"]:
            raise ValueError("Quality regressor changed since calibration")
        model = joblib.load(root / "quality_regressor.joblib")
        scores = np.clip(model.predict(features[config["features"]]), 0, 1)
    else:
        scores = models[config["scorer"]].predict_proba(features[config["features"]])[:, 1]
    best = int(np.argmax(scores))
    box = boxes[best] if scores[best] >= config["threshold"] else (0, 0, 0, 0)
    result = dict(
        box=output_box(box),
        max_score=float(scores[best]),
        threshold=config["threshold"],
        forward_counts=engine.counts,
    )
    base_write_json(root / "inference.json", result)
    print(json.dumps(result), flush=True)


def failure_gallery(predictions, annotations, args, root):
    from src.data import Rectangle, plt

    failures = predictions.sort_values(["eval_official_score", "id"]).head(6)
    images = [base_read_image(args.data_dir, ident) for ident in failures.id]
    maps = [base_residual_maps(image) for image in images]
    vmax = max(float(np.abs(r).max()) for group in maps for r in group.values()) or 1
    fig, axes = plt.subplots(
        len(images), 3, figsize=(10, 3 * len(images)), squeeze=False, layout="constrained"
    )
    for i, (_, row) in enumerate(failures.iterrows()):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"{row.id}\nscore={row.eval_official_score:.3f}", fontsize=10)
        for j, s in enumerate(base_SCALES, 1):
            im = axes[i, j].imshow(maps[i][s].mean(axis=2), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[i, j].set_title(f"Residual sigma={s}", fontsize=10)
        true = annotations[row.id]["box"]
        predicted = tuple(row["output_" + c] for c in base_COORDS)
        for ax in axes[i]:
            for box, color in ((true, "red"), (predicted, "gold")):
                if area(box):
                    x, y, xx, yy = box
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            xx - x,
                            yy - y,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.5,
                        )
                    )
            ax.axis("off")
    fig.suptitle("Worst holdout scores. Red: true attack; gold: output. Shared residual scale.")
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=0.5)
    fig.savefig(root / "proposal_failures.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(args):
    self_test()
    if args.image is not None:
        if args.experiment != "proposals":
            raise ValueError("--image is supported only by --experiment proposals")
        infer_image(args)
        return
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    rows, columns, models, manifest, annotations = prepare_reference(args, root)
    selection = selected_images(manifest, annotations, args.limit, args.seed)
    signature = dict(
        selection=selection,
        candidate_budget=args.candidate_budget,
        seed=args.seed,
        labels_sha256=base_digest(args.data_dir / "train_labels.csv"),
    )
    if (root / "selection.json").exists() and json.loads(
        (root / "selection.json").read_text()
    ) != signature:
        raise ValueError("Output belongs to another subset/budget; choose a new output directory")
    base_write_json(root / "selection.json", signature)
    modes = (
        ("robustness", "background", "ablation", "proposals")
        if args.experiment == "localization-all"
        else (args.experiment,)
    )
    engine = CandidateScorer(args, root / "cache") if any(m != "ablation" for m in modes) else None
    results = {}
    for mode in modes:
        print(
            f"Starting {mode}: selected image counts "
            + str({k: len(v) for k, v in selection.items()}),
            flush=True,
        )
        start = time.perf_counter()
        before = (
            engine.counts.copy()
            if engine
            else dict(
                original_forward_images=0,
                intervention_forward_images=0,
                forward_batches=0,
                cached_candidates=0,
                new_candidates=0,
            )
        )
        result = {
            "robustness": robustness,
            "background": background,
            "ablation": ablation,
            "proposals": proposals,
        }[mode](args, root, rows, columns, models, manifest, annotations, selection, engine)
        result["runtime_seconds"] = time.perf_counter() - start
        result["forward_counts"] = {
            k: engine.counts[k] - v if engine else 0 for k, v in before.items()
        }
        result["device"] = engine.device if engine else "CPU cached-feature replay"
        results[mode] = result
        base_write_json(root / f"{mode}_metrics.json", result)
        print(
            f"Completed {mode} in {result['runtime_seconds']:.1f}s: {result['forward_counts']}",
            flush=True,
        )
    for mode in ("robustness", "background", "ablation", "proposals"):
        if mode not in results and (root / f"{mode}_metrics.json").exists():
            results[mode] = json.loads((root / f"{mode}_metrics.json").read_text())
    write_report(args, root, selection, annotations, results)
