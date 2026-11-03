"""Baseline for patch analysis experiments."""

import csv
import json
from importlib.metadata import version

import numpy as np
import pandas as pd
from PIL import Image

from src.data import (
    COLORS,
    COORDS,
    SCALES,
    Rectangle,
    control_box,
    discover_split,
    features,
    parameters,
    parse_comment,
    plt,
    residual_maps,
    write_json,
)
from src.metrics import evaluate


def plots(df, gallery, output):
    columns = [c for c in df if c.startswith(("s0.7_", "s1.2_"))]
    for scale in SCALES:
        cols = [c for c in columns if c.startswith(f"s{scale}_")]
        fig, axes = plt.subplots(4, 6, figsize=(18, 11))
        for ax, col in zip(axes.flat, cols):
            groups = [df.loc[df.kind == k, col] for k in COLORS]
            ax.boxplot(
                groups,
                tick_labels=[f"{k}\nn={len(g)}" for k, g in zip(COLORS, groups)],
                showfliers=False,
            )
            ax.set_title(col, fontsize=9)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(cols) :]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(output / f"features_{scale}.png", dpi=140)
        plt.close(fig)
    if gallery:
        fig, axes = plt.subplots(
            len(gallery), 3, figsize=(10, 3 * len(gallery)), squeeze=False, layout="constrained"
        )
        vmax = max(float(np.abs(r).max()) for _, _, maps, _ in gallery for r in maps.values()) or 1
        for row, (ident, image, maps, records) in enumerate(gallery):
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(ident)
            for col, (s, r) in enumerate(maps.items(), 1):
                im = axes[row, col].imshow(r.mean(axis=2), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                axes[row, col].set_title(f"sigma={s}\nsigned RGB mean", fontsize=10)
            for ax in axes[row]:
                for record in records:
                    x, y, xx, yy = record["box"]
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            xx - x,
                            yy - y,
                            fill=False,
                            edgecolor=COLORS[record["kind"]],
                            linewidth=1,
                        )
                    )
                ax.axis("off")
        fig.suptitle("Red: adversarial; cyan: decoy; green: background. Shared residual scale.")
        fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=0.5)
        fig.savefig(output / "gallery.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def run_baseline(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / "train_labels.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = rows[: args.limit]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate image IDs")
    parsed, issues, dimensions = [], [], set()
    for row in rows:
        try:
            if row["split"] != "train":
                raise ValueError("non-training row")
            regions, meta = parse_comment(row["comment"], tuple(int(row[c]) for c in COORDS))
            parsed.append((row, regions, meta))
            dimensions.update(
                (r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) for r in regions
            )
        except (ValueError, KeyError) as exc:
            issues.append(dict(id=row["id"], error=str(exc), comment=row["comment"]))
    (output / "annotation_issues.json").write_text(json.dumps(issues, indent=2))
    if issues:
        raise ValueError(
            f"{len(issues)} unsupported/inconsistent annotations; see annotation_issues.json. No partial benchmark produced."
        )
    rng = np.random.default_rng(args.seed)
    records, gallery, failed_controls = [], [], []
    for i, (row, regions, meta) in enumerate(parsed):
        with Image.open(args.data_dir / "train" / "images" / (row["id"] + ".png")) as source:
            if source.mode != "RGB" or source.size != (64, 64):
                raise ValueError(f"invalid image mode/size: {row['id']}")
            image = np.asarray(source, dtype=np.float64) / 255
        maps = residual_maps(image)
        control = control_box([r["box"] for r in regions], sorted(dimensions), rng)
        if control is None:
            failed_controls.append(row["id"])
        else:
            regions = regions + [
                dict(kind="background", region_type="control", box=control, **parameters(""))
            ]
        for number, region in enumerate(regions):
            record = {
                **meta,
                **region,
                "id": row["id"],
                "region_number": number,
                "comment": row["comment"],
            }
            record.update(zip(COORDS, record.pop("box")))
            record.update(features(maps, region["box"]))
            records.append(record)
        if len(gallery) < 6:
            gallery.append((row["id"], image, maps, regions))
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(parsed)} images", flush=True)
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("no annotated dimensions or regions available")
    for col in ("epsilon", "l2", "reported_linf"):
        bins = (
            [0, 0.5, 1, 2, np.inf]
            if col == "l2"
            else [0, 4 / 255, 8 / 255, 12 / 255, 20 / 255, np.inf]
        )
        df[col + "_range"] = (
            pd.cut(df[col], bins, include_lowest=True).astype("string").fillna("missing")
        )
    metrics = dict(
        seed=args.seed,
        images=len(rows),
        annotation_issues=len(issues),
        control_failures=failed_controls,
        versions={
            p: version(p)
            for p in ("numpy", "pandas", "Pillow", "scipy", "scikit-learn", "matplotlib")
        },
        region_counts={k: int(v) for k, v in df.kind.value_counts().items()},
        evaluation=evaluate(df, [r["id"] for r in rows], args.seed, discover_split(args)),
    )
    if "train_image_ids" in metrics["evaluation"]:
        write_json(
            output / "split_manifest.json",
            {k: metrics["evaluation"][k] for k in ("train_image_ids", "validation_image_ids")},
        )
    df.to_csv(output / "regions.csv", index=False)
    for key in (
        "attack_family",
        "jpeg_quality",
        "region_type",
        "epsilon_range",
        "l2_range",
        "reported_linf_range",
    ):
        df.groupby(["kind", key], dropna=False).size().rename("count").to_csv(
            output / f"counts_{key}.csv"
        )
    feat_cols = [c for c in df if c.startswith(("s0.7_", "s1.2_"))]
    df.groupby("kind")[feat_cols].agg(["count", "mean", "std", "median"]).to_csv(
        output / "feature_summary.csv"
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    plots(df, gallery, output)
    lines = [
        "# Known-box patch baseline",
        "",
        f"Analyzed {len(rows)} training images; seed {args.seed}.",
        f"Region counts: {metrics['region_counts']}. Unplaceable controls: {len(failed_controls)}. Annotation issues: {len(issues)}.",
        "",
        "## Held-out results",
        "",
        "Class 1 is adversarial; confusion matrices use rows=true, columns=predicted, labels=[decoy, adversarial].",
        "",
    ]
    for name in ("energy", "residual"):
        if name in metrics["evaluation"]:
            m = metrics["evaluation"][name]["overall"]
            lines.append(
                f"- {name}: ROC-AUC {m['roc_auc']:.4f}; AP {m['average_precision']:.4f}; confusion matrix {m['confusion_matrix']}; strict ranking {m['strict_ranking_accuracy']} over {m['ranking_images']} images."
            )
    if "residual" in metrics["evaluation"]:
        full = metrics["evaluation"]["residual"]
        delta = full["overall"]["roc_auc"] - metrics["evaluation"]["energy"]["overall"]["roc_auc"]
        lines += [
            "",
            f"Full-feature minus energy ROC-AUC: {delta:+.4f}. This is descriptive evidence from one holdout, not a significance test.",
            "Largest absolute standardized model coefficients (correlated predictors; not causal importance):",
        ]
        for feature, coefficient in sorted(
            full["standardized_coefficients"].items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:8]:
            lines.append(f"- {feature}: {coefficient:+.4f}")
        lines += ["", "## Diagnostic groups", ""]
        for key, groups in full["breakdowns"].items():
            lines.append(f"### {key}")
            for label, m in groups.items():
                lines.append(
                    f"- {label}: attack={m['adversarial']}, decoy={m['decoy']}, AUC={m['roc_auc']}; "
                    + ("too small or single-class" if m["small_group"] else "at least 20 per class")
                )
            if key.endswith("_range"):
                comparable = [
                    k for k, m in groups.items() if k != "missing" and not m["small_group"]
                ]
                lines.append(
                    f"Comparable recorded-strength bins with at least 20 per class: {len(comparable)}. These are diagnostic bins, not matched perturbations."
                )
    if "unavailable" in metrics["evaluation"]:
        lines.append("Evaluation unavailable: " + metrics["evaluation"]["unavailable"])
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "Energy uses only overall residual RMS at the two scales. The full model uses 44 residual features. All scaling is fitted on training images only; background controls are excluded from classifiers. Image IDs are disjoint across the 80/20 holdout.",
        "Breakdowns in metrics.json include attack family (decoys grouped by their source image attack), JPEG quality, and separate epsilon, L2, and reported-Linf ranges. Null discrimination metrics indicate a single-class group. Groups with fewer than 20 examples of either class are flagged small_group; these are too small for reliable comparable-strength claims. Missing strengths are not zero. Nominal epsilon and reported Linf are not interchangeable.",
        "Filtering residuals contain natural texture, not the true perturbation. Energy differences are not evidence of adversarial structure. Even equal nominal-strength bins do not match image texture or realized perturbation energy. Controls are sampled from available nonoverlapping space using observed dimensions and may have different spatial/content distributions.",
        "These metrics measure known-box classification, not localization or competition score. No test images were read. Hyperparameters are fixed (C=1, threshold=0.5). Only one holdout is evaluated; no uncertainty intervals or tuning are claimed.",
        "",
        "![Residual features](features_0.7.png)",
        "![Residual features](features_1.2.png)",
        "![Gallery](gallery.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                k: v["overall"]
                for k, v in metrics["evaluation"].items()
                if k in ("energy", "residual")
            },
            indent=2,
        )
    )
