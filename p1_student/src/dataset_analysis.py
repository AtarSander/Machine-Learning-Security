"""Descriptive patch geography and image-only attack-versus-decoy comparison."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import laplace
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src import data as base
from src.metrics import experiment_metrics


def pixel_features(image, box):
    x, y, xx, yy = box
    crop = image[y:yy, x:xx]
    lum = image @ np.asarray([0.2126, 0.7152, 0.0722])
    patch = lum[y:yy, x:xx]
    gy, gx = np.gradient(lum)
    result = {}
    for j, c in enumerate("rgb"):
        result[f"pixel_{c}_mean"] = float(crop[:, :, j].mean())
        result[f"pixel_{c}_std"] = float(crop[:, :, j].std())
    hist = np.histogram(patch, bins=32, range=(0, 1))[0].astype(float)
    hist /= hist.sum()
    result.update(
        pixel_luminance_mean=float(patch.mean()),
        pixel_luminance_std=float(patch.std()),
        pixel_contrast_90=float(np.quantile(patch, 0.95) - np.quantile(patch, 0.05)),
        pixel_entropy=float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()),
        pixel_gradient_rms=float(np.sqrt((gx[y:yy, x:xx] ** 2 + gy[y:yy, x:xx] ** 2).mean())),
        pixel_laplacian_rms=float(np.sqrt((laplace(lum, mode="reflect")[y:yy, x:xx] ** 2).mean())),
        pixel_horizontal_corr=base.correlation(patch[:, :-1], patch[:, 1:]),
        pixel_vertical_corr=base.correlation(patch[:-1], patch[1:]),
        pixel_saturation_mean=float((crop.max(2) - crop.min(2)).mean()),
    )
    for a, b in ((0, 1), (0, 2), (1, 2)):
        result[f"pixel_corr_{'rgb'[a]}{'rgb'[b]}"] = base.correlation(crop[:, :, a], crop[:, :, b])
    return result


def geography(annotations, root):
    maps = {k: np.zeros((64, 64), float) for k in ("adversarial", "decoy", "union")}
    rows = []
    for ident, a in annotations.items():
        local = {k: np.zeros((64, 64), bool) for k in ("adversarial", "decoy")}
        for i, r in enumerate(a["regions"]):
            x, y, xx, yy = r["box"]
            local[r["kind"]][y:yy, x:xx] = True
            rows.append(
                dict(
                    id=ident,
                    region_number=i,
                    kind=r["kind"],
                    region_type=r["region_type"],
                    x_min=x,
                    y_min=y,
                    x_max=xx,
                    y_max=yy,
                    width=xx - x,
                    height=yy - y,
                    area=(xx - x) * (yy - y),
                    center_x=(x + xx) / 2,
                    center_y=(y + yy) / 2,
                )
            )
        for k in local:
            maps[k] += local[k]
        maps["union"] += local["adversarial"] | local["decoy"]
    rows = pd.DataFrame(rows)
    rows.to_csv(root / "patch_geometry.csv", index=False)
    summary = {}
    for kind, g in rows.groupby("kind"):
        summary[kind] = dict(
            patches=len(g),
            images=int(g.id.nunique()),
            entirely_bottom_half=int((g.y_min >= 32).sum()),
            entirely_top_half=int((g.y_max <= 32).sum()),
            cross_midline=int(((g.y_min < 32) & (g.y_max > 32)).sum()),
            touch_bottom=int((g.y_max == 64).sum()),
            touch_left=int((g.x_min == 0).sum()),
            touch_right=int((g.x_max == 64).sum()),
            statistics={
                c: dict(
                    mean=float(g[c].mean()),
                    std=float(g[c].std()),
                    min=float(g[c].min()),
                    q25=float(g[c].quantile(0.25)),
                    median=float(g[c].median()),
                    q75=float(g[c].quantile(0.75)),
                    max=float(g[c].max()),
                )
                for c in (
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "center_x",
                    "center_y",
                    "width",
                    "height",
                    "area",
                )
            },
        )
    for k in maps:
        maps[k] /= len(annotations)
    np.savez_compressed(root / "occupancy_maps.npz", **maps)
    base.write_json(
        root / "spatial_metrics.json",
        dict(
            images=len(annotations),
            groups=summary,
            normalization="Per-pixel fraction of all labeled images; decoys are unioned per image to avoid double counting",
            average_union_fraction=float(maps["union"].mean()),
            region_type_counts=rows.region_type.value_counts().to_dict(),
        ),
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, k in zip(axes, maps):
        im = ax.imshow(maps[k] * 100, vmin=0, vmax=100, cmap="magma", extent=(0, 64, 64, 0))
        ax.axhline(32, color="cyan", linestyle="--", linewidth=1)
        ax.set_title(k)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes.tolist(), label="% of all 5000 images covering this pixel", shrink=0.8)
    fig.savefig(root / "spatial_occupancy.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for kind, g in rows.groupby("kind"):
        for ax, c, bins in zip(
            axes,
            ["center_x", "center_y", "area"],
            [np.arange(0, 65, 4), np.arange(0, 65, 4), np.arange(300, 701, 25)],
        ):
            ax.hist(g[c], bins=bins, density=True, histtype="step", linewidth=2, label=kind)
            ax.set_xlabel(c)
            ax.set_ylabel("Density")
    axes[0].legend()
    fig.savefig(root / "geometry_distributions.png", dpi=160)
    plt.close(fig)
    return rows, summary


def bootstrap_auc(table, probability):
    # Resample whole images, retaining every candidate from each sampled image.
    ids = table.id.unique()
    positions = {ident: np.flatnonzero(table.id.to_numpy() == ident) for ident in ids}
    rng = np.random.default_rng(42)
    y = (table.kind == "adversarial").to_numpy()
    p = np.asarray(probability)
    scores = []
    for _ in range(200):
        take = np.concatenate([positions[i] for i in rng.choice(ids, size=len(ids), replace=True)])
        scores.append(roc_auc_score(y[take], p[take]))
    return np.quantile(scores, [0.025, 0.975]).tolist()


def run(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    annotations = {}
    with (args.data_dir / "train_labels.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            box = tuple(int(row[c]) for c in base.COORDS)
            regions, meta = base.parse_comment(row["comment"], box)
            annotations[row["id"]] = dict(regions=regions, **meta)
    geometry, spatial = geography(annotations, root)
    old = pd.read_csv(
        args.baseline_dir / "regions.csv", dtype={"id": str}, float_precision="round_trip"
    )
    assert set(old.id) == set(annotations) and len(annotations) == 5000
    residual = base.residual_columns(old)
    rows = old[old.kind.isin(["adversarial", "decoy"])].copy()
    keys = ["id", "region_number"]
    assert len(rows) == len(geometry)
    merged = rows.merge(geometry, on=keys, suffixes=("", "_check"), validate="one_to_one")
    assert all((merged[c] == merged[c + "_check"]).all() for c in base.COORDS + ["kind"])
    raw_path = root / "pixel_features.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, dtype={"id": str}, float_precision="round_trip")
    else:
        extracted = []
        for index, (ident, g) in enumerate(rows.groupby("id", sort=False)):
            image = base.read_image(args.data_dir, ident)
            for _, r in g.iterrows():
                extracted.append(
                    dict(
                        id=ident,
                        region_number=int(r.region_number),
                        **pixel_features(image, tuple(int(r[c]) for c in base.COORDS)),
                    )
                )
            if (index + 1) % 1000 == 0:
                print(
                    f"Image features: {index + 1}/{rows.id.nunique()} patch-containing images",
                    flush=True,
                )
        raw = pd.DataFrame(extracted)
        raw.to_csv(raw_path, index=False)
    rows = rows.merge(raw, on=keys, validate="one_to_one")
    pixels = [c for c in raw if c.startswith("pixel_")]
    features = pixels + residual
    assert np.isfinite(rows[features]).all().all()
    train = rows[rows.partition == "train"]
    test = rows[rows.partition == "validation"]
    assert not set(train.id) & set(test.id)
    manifest = json.loads((args.baseline_dir / "metrics.json").read_text())["evaluation"]
    base.write_json(
        root / "split_manifest.json",
        {k: manifest[k] for k in ("train_image_ids", "validation_image_ids")},
    )
    assert set(train.id) <= set(manifest["train_image_ids"]) and set(test.id) <= set(
        manifest["validation_image_ids"]
    )
    stats = []
    for c in features:
        attack = rows.loc[rows.kind == "adversarial", c]
        decoy = rows.loc[rows.kind == "decoy", c]
        pooled = np.sqrt((attack.var() + decoy.var()) / 2)
        auc = roc_auc_score(train.kind == "adversarial", train[c])
        sign = 1 if auc >= 0.5 else -1
        stats.append(
            dict(
                feature=c,
                attack_mean=attack.mean(),
                decoy_mean=decoy.mean(),
                attack_median=attack.median(),
                decoy_median=decoy.median(),
                standardized_mean_difference=(attack.mean() - decoy.mean()) / pooled
                if pooled
                else 0,
                train_direction=sign,
                train_univariate_auc=max(auc, 1 - auc),
                holdout_univariate_auc=roc_auc_score(test.kind == "adversarial", sign * test[c]),
            )
        )
    stats = pd.DataFrame(stats).sort_values("train_univariate_auc", ascending=False)
    stats.to_csv(root / "feature_comparison.csv", index=False)
    results = {}
    for name, columns in [
        ("pixels_only", pixels),
        ("residual_only", residual),
        ("pixels_and_residual", features),
    ]:
        for family in ("logistic", "boosting"):
            if family == "logistic":
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42)
                )
            else:
                model = HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=42,
                )
            model.fit(train[columns], train.kind == "adversarial")
            prediction = model.predict_proba(test[columns])[:, 1]
            scored = test.copy()
            scored["probability"] = prediction
            m = experiment_metrics(scored, "probability")
            m["auc_image_bootstrap_95_interval"] = bootstrap_auc(scored, prediction)
            m["features"] = columns
            results[name + "_" + family] = m
            scored[["id", "region_number", "kind", "probability"]].to_csv(
                root / (name + "_" + family + "_predictions.csv"), index=False
            )
            print(name, family, m["overall"], flush=True)
    # Same-image paired differences control image content better than pooled summaries.
    paired = []
    for ident, g in rows.groupby("id"):
        if set(g.kind) != {"adversarial", "decoy"}:
            continue
        a = g[g.kind == "adversarial"].iloc[0]
        for _, d in g[g.kind == "decoy"].iterrows():
            paired.append(
                dict(
                    id=ident, decoy_type=d.region_type, **{c: float(a[c] - d[c]) for c in features}
                )
            )
    paired = pd.DataFrame(paired)
    paired.to_csv(root / "same_image_feature_differences.csv", index=False)
    selected = [
        "pixel_luminance_std",
        "pixel_gradient_rms",
        "pixel_entropy",
        "s0.7_rms_all",
        "s1.2_rms_all",
        "s0.7_corr_rg",
    ]
    paired_summary = {
        c: dict(
            pairs=len(paired),
            images=int(paired.id.nunique()),
            attack_greater_fraction=float((paired[c] > 0).mean()),
            median_difference=float(paired[c].median()),
        )
        for c in selected
    }
    # Include decoy-type property distributions without treating type labels as predictors.
    summaries = rows.groupby(["kind", "region_type"])[selected].agg(["mean", "median", "std"])
    summaries.to_csv(root / "properties_by_patch_type.csv")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, c in zip(axes.ravel(), selected):
        lo, hi = rows[c].quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 35)
        for kind, g in rows.groupby("kind"):
            ax.hist(g[c], bins=bins, density=True, histtype="step", linewidth=1.5, label=kind)
        ax.set_title(c)
        ax.set_ylabel("Density")
    axes[0, 0].legend()
    fig.suptitle("All labeled patches; central 99% of each feature shown")
    fig.savefig(root / "image_feature_distributions.png", dpi=160)
    plt.close(fig)
    metadata = dict(
        images=len(annotations),
        image_groups=pd.Series([a["image_kind"] for a in annotations.values()])
        .value_counts()
        .to_dict(),
        patches=len(rows),
        attack_patches=int((rows.kind == "adversarial").sum()),
        decoy_patches=int((rows.kind == "decoy").sum()),
        training_regions=len(train),
        holdout_regions=len(test),
        features=dict(pixels=pixels, residual=residual),
        spatial=spatial,
        models=results,
        paired_differences=paired_summary,
        runtime_seconds=time.perf_counter() - start,
        labels_sha256=base.digest(args.data_dir / "train_labels.csv"),
        cached_residuals_sha256=base.digest(args.baseline_dir / "regions.csv"),
        scope="All 5000 labeled training images for descriptions. Original 4000/1000 image split for model evaluation. Previously inspected holdout: exploratory comparison, not fresh validation.",
        excluded_predictors="No coordinates, box size, labels, type, class, epsilon, JPEG metadata, ResNet features/logits or localizer outputs.",
        models_configuration=dict(
            logistic="StandardScaler + LogisticRegression C1 max_iter2000 seed42",
            boosting="HistGradientBoosting lr.05 200 iterations 15 leaves min_leaf20 L2=1 early_stopping=False seed42",
        ),
    )
    base.write_json(root / "metrics.json", metadata)
    lines = [
        "# Whole labeled-dataset patch analysis",
        "",
        metadata["scope"],
        "",
        "“Fake” means annotated decoys; “attack” means annotated adversarial patches. Clean original counterparts are unavailable. These image statistics mix scene content, JPEG effects and added noise.",
        "",
        "## Spatial distribution",
        "",
        f"{len(annotations)} images; {len(rows)} regions: 4000 attacks and 4746 decoys.",
        "",
        "| Type | Regions | Entirely y>=32 | Crossing y=32 | Mean center (x,y) | Mean width × height | Mean area |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for kind, s in spatial.items():
        q = s["statistics"]
        lines.append(
            f"| {kind} | {s['patches']} | {s['entirely_bottom_half']} | {s['cross_midline']} | ({q['center_x']['mean']:.2f}, {q['center_y']['mean']:.2f}) | {q['width']['mean']:.2f} × {q['height']['mean']:.2f} | {q['area']['mean']:.2f} |"
        )
    lines += [
        "",
        "Occupancy maps show the fraction of all 5000 images covering each pixel; multiple decoys in one image are unioned, not counted twice.",
        "![Spatial occupancy](spatial_occupancy.png)",
        "![Patch geometry](geometry_distributions.png)",
        "",
        "## Distinguishability from image features only",
        "",
        metadata["excluded_predictors"],
        f"{len(pixels)} raw color/texture features, 44 existing full-image filtering-residual features. Geometry is descriptive only. Fixed configurations, no hyperparameter search.",
        "",
        "| Features/model | AUC | Image-bootstrap 95% interval | AP | Within-image ranking |",
        "|---|---:|---|---:|---:|",
    ]
    for name, m in results.items():
        o = m["overall"]
        ci = m["auc_image_bootstrap_95_interval"]
        lines.append(
            f"| {name} | {o['roc_auc']:.4f} | {ci[0]:.4f}–{ci[1]:.4f} | {o['average_precision']:.4f} | {o['strict_ranking_accuracy']:.4f} |"
        )
    o = next(iter(results.values()))["overall"]
    lines += [
        "",
        f"Holdout: {o['adversarial']} attacks and {o['decoy']} decoys; AP prevalence reference {o['random_average_precision']:.4f}. Ranking on {o['ranking_images']} eligible images; random-choice reference {o['random_ranking_accuracy']:.4f}. Bootstrap intervals resample image IDs, not individual regions, and do not capture model selection or dataset shift.",
        "",
        "![Image feature distributions](image_feature_distributions.png)",
        "",
        "## Property summaries across all labeled patches",
        "",
        "| Feature | Attack mean | Decoy mean | Attack median | Decoy median | Standardized difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in selected:
        r = stats.set_index("feature").loc[c]
        lines.append(
            f"| {c} | {r.attack_mean:.5f} | {r.decoy_mean:.5f} | {r.attack_median:.5f} | {r.decoy_median:.5f} | {r.standardized_mean_difference:.3f} |"
        )
    lines += [
        "",
        "## Same-image comparisons",
        "",
        "| Feature | Attack > decoy fraction | Median paired difference |",
        "|---|---:|---:|",
    ]
    for c, r in paired_summary.items():
        lines.append(f"| {c} | {r['attack_greater_fraction']:.4f} | {r['median_difference']:.5f} |")
    lines += [
        "",
        f"{len(paired)} attack/decoy pairs from {paired.id.nunique()} images. Each attack is paired with every decoy in that image, so pairs are correlated. No hypothesis-test significance is claimed.",
        "",
        "Complete geometry quantiles, per-type image properties, training-oriented univariate AUCs, and model subgroup counts are in the CSV/JSON artifacts. "
        "Residual magnitude is not measured perturbation strength; no clean counterpart exists. These tabular results are not an upper bound on what a learned pixel model could achieve. "
        "No ResNet or U-Net is run, and no submission or test images are used.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--baseline-dir", type=Path, default=Path("outputs/full"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_analysis"))
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)
