"""Experiments for patch analysis experiments."""

import argparse
import csv
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.baseline import run_baseline
from src.checks import baseline_checks as self_test
from src.checks import extended_checks
from src.classifiers import (
    cnn_experiment,
    intervention_measurements,
    oracle_summary,
    tabular_experiment,
)
from src.data import (
    COORDS,
    SCALES,
    baseline_source,
    box_valid,
    digest,
    discover_split,
    parse_comment,
    residual_columns,
    write_json,
)
from src.metrics import score_metrics
from src.reports import experiment_report


def run_experiments(args):
    start = time.perf_counter()
    source = baseline_source(args)
    if source is None:
        run_baseline(args)
        source = args.output_dir
    baseline_path = source / "regions.csv"
    df = pd.read_csv(baseline_path, dtype={"id": str})
    manifest = discover_split(args)
    with (args.data_dir / "train_labels.csv").open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    selected_ids = [r["id"] for r in labels][: args.limit]
    if not set(selected_ids) <= set(df.id):
        raise ValueError(
            "Baseline records do not cover requested images; rerun baseline without --limit in a new output directory"
        )
    df = df[df.id.isin(selected_ids)].copy().reset_index(drop=True)
    if df.duplicated(["id", "region_number"]).any():
        raise ValueError("Duplicate region keys")
    expected_partition = {
        ident: part for part in ("train", "validation") for ident in manifest[part + "_image_ids"]
    }
    if not df.partition.equals(df.id.map(expected_partition)):
        raise ValueError("Saved regions disagree with split manifest")
    if not np.isfinite(df[residual_columns(df)]).all().all():
        raise ValueError("Nonfinite baseline residual features")
    lookup = {r["id"]: r for r in labels}
    for ident, group in df.groupby("id", sort=False):
        row = lookup[ident]
        regions, _ = parse_comment(row["comment"], tuple(int(row[c]) for c in COORDS))
        annotated = group[group.kind != "background"]
        if len(annotated) != len(regions) or not (group.comment == row["comment"]).all():
            raise ValueError(f"Stale baseline annotations: {ident}")
        for record, (_, saved) in zip(regions, annotated.iterrows()):
            if record["kind"] != saved.kind or tuple(record["box"]) != tuple(
                saved[c] for c in COORDS
            ):
                raise ValueError(f"Stale baseline box: {ident}")
        for saved in group.itertuples():
            box_valid((saved.x_min, saved.y_min, saved.x_max, saved.y_max))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "experiments"
    output.mkdir(exist_ok=True)
    write_json(args.output_dir / "split_manifest.json", manifest)
    selection = dict(
        image_ids=selected_ids,
        region_keys=df[["id", "region_number"]].values.tolist(),
        baseline_regions_sha256=digest(baseline_path),
        labels_sha256=digest(args.data_dir / "train_labels.csv"),
        seed=args.seed,
        selection="first whole images in CSV order; persisted baseline partitions",
        baseline_directory=str(source.resolve()),
    )
    previous = output / "selection.json"
    if previous.exists() and json.loads(previous.read_text()) != selection:
        raise ValueError(
            "Output directory belongs to another sample selection; choose a new --output-dir"
        )
    write_json(previous, selection)
    write_json(
        output / "versions.json",
        {
            p: version(p)
            for p in ("numpy", "pandas", "Pillow", "scipy", "scikit-learn", "matplotlib")
        },
    )
    extended_checks(args)
    binary = df[df.kind != "background"]
    if any(binary[binary.partition == p].kind.nunique() != 2 for p in ("train", "validation")):
        write_json(
            output / "unavailable.json",
            {"reason": "Both classes required in training and validation"},
        )
        (args.output_dir / "report.md").write_text(
            "# Evaluation unavailable\n\nBoth target classes must exist in each persisted partition.\n"
        )
        return
    results = {}
    for name, cols in [
        ("energy", [f"s{s}_rms_all" for s in SCALES]),
        ("residual", residual_columns(df)),
    ]:
        results[name] = tabular_experiment(name, df, cols, args, output)
        predicted = pd.read_csv(output / f"{name}_predictions.csv")
        saved = df[(df.partition == "validation") & (df.kind != "background")][
            name + "_probability"
        ].to_numpy()
        results[name]["max_difference_from_persisted_predictions"] = float(
            np.max(np.abs(predicted.probability - saved))
        )
        persisted_frame = df[(df.partition == "validation") & (df.kind != "background")].copy()
        results[name]["persisted_reference_on_selected_holdout"] = score_metrics(
            persisted_frame, name + "_probability"
        )
        results[name]["comparison_note"] = (
            "Subset runs refit references on selected training images; differences from full baseline are expected."
            if args.limit
            else "Full-data reference refitted on the persisted split; numerical discrepancies are reported explicitly."
        )
        write_json(output / f"{name}_metrics.json", results[name])
    # Publish the initial diagnosis before fitting any new model.
    write_json(output / "baseline_diagnosis.json", results["residual"])
    if args.experiment in ("boosting", "all"):
        results["boosting"] = tabular_experiment("boosting", df, residual_columns(df), args, output)
    if args.experiment in ("cnn", "all"):
        results["cnn"] = cnn_experiment(df, args, output)
    oracle = None
    if args.experiment in ("resnet", "all"):
        measured, config = intervention_measurements(df, args, output)
        oracle = oracle_summary(measured)
        write_json(output / "resnet_oracle_diagnostics.json", oracle)
        extra = []
        augmented = df.copy()
        for scale in SCALES:
            values = measured[measured.scale == scale].set_index(["id", "region_number"])
            for key in (
                "predicted_margin_drop",
                "predicted_probability_change",
                "probability_l2_distance",
            ):
                column = f"intervention_{scale}_{key}"
                extra.append(column)
                augmented[column] = (
                    values[key]
                    .reindex(pd.MultiIndex.from_frame(df[["id", "region_number"]]))
                    .to_numpy()
                )
        # Decide whether to fit the label-free model using training diagnostics only.
        helpful = any(
            v["diagnostic_only_discrimination"]
            and v["diagnostic_only_discrimination"]["roc_auc"] > 0.5
            and (v["within_image"]["decoy_all"]["fraction_attack_larger_gain"] or 0) > 0.5
            for v in oracle["train"].values()
        )
        config["label_free_gate"] = dict(
            rule="Training oracle AUC > 0.5 and paired gain win fraction > 0.5 at either scale",
            passed=helpful,
        )
        write_json(output / "resnet_config.json", config)
        if helpful:
            results["resnet_label_free"] = tabular_experiment(
                "resnet_label_free", augmented, residual_columns(df) + extra, args, output
            )
    # Include completed experiments from prior invocations only on this identical selection.
    for name in ("boosting", "cnn", "resnet_label_free"):
        path = output / f"{name}_metrics.json"
        if name not in results and path.exists():
            results[name] = json.loads(path.read_text())
    if oracle is None and (output / "resnet_oracle_diagnostics.json").exists():
        oracle = json.loads((output / "resnet_oracle_diagnostics.json").read_text())
    experiment_report(df, results, oracle, args, output, time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=[
            "baseline",
            "boosting",
            "cnn",
            "resnet",
            "all",
            "robustness",
            "background",
            "ablation",
            "proposals",
            "localization-all",
            "localizer-train",
            "localizer-eval",
            "localizer-pipeline",
            "localizer-all",
        ],
        default="baseline",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Existing baseline results; otherwise discover outputs/full",
    )
    parser.add_argument(
        "--epochs", type=int, default=8, help="Fixed CNN training budget; no holdout selection"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--reference-dir", type=Path, default=Path("outputs/next/experiments"))
    parser.add_argument("--candidate-budget", type=int, default=16)
    parser.add_argument(
        "--image", type=Path, help="Annotation-free inference with a saved proposal pipeline"
    )
    parser.add_argument(
        "--localizer-source",
        type=Path,
        default=Path("outputs/localization"),
        help="Persisted split and frozen scorers",
    )
    parser.add_argument("--localizer-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Train the localizer with random flips/transposes/rotations and select over all orientations",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume saved localizer training state"
    )
    args = parser.parse_args()
    self_test()
    if args.self_test:
        if args.experiment != "baseline":
            extended_checks(args)
        if args.experiment in (
            "robustness",
            "background",
            "ablation",
            "proposals",
            "localization-all",
        ):
            from src.checks import localization_checks as localization_tests

            localization_tests()
        if args.experiment.startswith("localizer-"):
            from src.noise_model import self_test as localizer_tests

            localizer_tests(args)
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")
    if args.candidate_budget < 1:
        parser.error("--candidate-budget must be positive")
    if not 1 <= args.localizer_epochs <= 30 or args.batch_size < 1:
        parser.error("--localizer-epochs must be 1..30 and --batch-size positive")
    with threadpool_limits(limits=args.threads):
        if args.experiment.startswith("localizer-"):
            from src.noise_pipeline import run

            run(args)
        elif args.experiment in (
            "robustness",
            "background",
            "ablation",
            "proposals",
            "localization-all",
        ):
            from src.localization import run

            run(args)
        elif args.experiment != "baseline":
            run_experiments(args)
        else:
            run_baseline(args)
