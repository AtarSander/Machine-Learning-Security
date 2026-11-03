"""Reports for patch analysis experiments."""

import json
import shutil

import numpy as np
import pandas as pd

from src.baseline import plots
from src.data import SIZES, read_image, residual_maps, write_json


def experiment_report(df, results, oracle, args, output, seconds):
    def fmt(value):
        return "unavailable" if value is None else f"{value:.4f}"

    lines = [
        "# Known-box classification experiments",
        "",
        f"{df.id.nunique()} source images; persisted split, seed {args.seed}. Latest invocation runtime: {seconds:.1f}s.",
        "All models below use the identical held-out adversarial/decoy region keys. Background controls are excluded. Confusion matrices: rows=true, columns=predicted, order=[decoy, attack].",
        "",
        "| Model | AUC | AP | Strict image ranking | Confusion matrix | Fit/evaluation seconds |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name, result in results.items():
        m = result["overall"]
        lines.append(
            f"| {name} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | {fmt(m['strict_ranking_accuracy'])} | {m['confusion_matrix']} | {result['runtime_seconds']:.1f} |"
        )
    m = results["residual"]["overall"]
    lines += [
        "",
        f"Held-out regions: {m['n']} ({m['adversarial']} attacks, {m['decoy']} decoys); ranking images: {m['ranking_images']}.",
        f"Random AP reference (attack prevalence): {fmt(m['random_average_precision'])}. Random within-image choice reference: {fmt(m['random_ranking_accuracy'])}.",
        "",
        "## Baseline reproduction and diagnosis",
        "",
    ]
    for name in ("energy", "residual"):
        lines.append(
            f"- {name}: maximum probability difference from saved baseline = {results[name]['max_difference_from_persisted_predictions']:.3g}. {results[name]['comparison_note']}"
        )
        if "persisted_reference_on_selected_holdout" in results[name]:
            previous = results[name]["persisted_reference_on_selected_holdout"]
            lines.append(
                f"  Persisted AUC/AP: {fmt(previous['roc_auc'])}/{fmt(previous['average_precision'])}; rerun AUC/AP: {fmt(results[name]['overall']['roc_auc'])}/{fmt(results[name]['overall']['average_precision'])}."
            )
    family_scores = results["residual"]["subgroups"]["attack_family"]
    measured_families = {k: v for k, v in family_scores.items() if v["roc_auc"] is not None}
    lines.append(
        "Measured baseline family AUCs: "
        + ", ".join(f"{k}={v['roc_auc']:.4f}" for k, v in measured_families.items())
        + ". No family is excluded from the benchmark."
    )
    lines.append("")
    for key in (
        "jpeg_quality",
        "source_attack_epsilon_range",
        "source_attack_l2_range",
        "paired_decoy_type",
    ):
        groups = {
            k: v
            for k, v in results["residual"]["subgroups"].get(key, {}).items()
            if k != "missing" and v["roc_auc"] is not None and not v["small_group"]
        }
        if groups:
            hardest = min(groups, key=lambda k: groups[k]["roc_auc"])
            easiest = max(groups, key=lambda k: groups[k]["roc_auc"])
            lines.append(
                f"- {key}: lowest AUC {groups[hardest]['roc_auc']:.4f} in {hardest}; highest {groups[easiest]['roc_auc']:.4f} in {easiest}. Counts below; these are descriptive associations, not causal effects."
            )
    lines += [
        "",
        "Strength subgroups select images by their attack strength and include the attack and all comparison decoys from those images. Paired decoy-type groups select attacks from images containing that type and only those decoys. Same-image/same-strength bins retain only matching attack/decoy pairs; epsilon, L2 and reported Linf remain separate. Groups overlap and are not independent.",
        "",
        "## Subgroups",
        "",
        "| Model | Definition | Group | Attacks | Decoys | AUC | AP | Ranking | Small group |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, result in results.items():
        for key, groups in result["subgroups"].items():
            for label, m in groups.items():
                lines.append(
                    f"| {name} | {key} | {label} | {m['adversarial']} | {m['decoy']} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | {fmt(m['strict_ranking_accuracy'])} | {m['small_group']} |"
                )
    lines += [
        "",
        "Small means fewer than 20 examples of either class. Single-class discrimination is unavailable. Equal annotation bins do not imply equal realized perturbations or residual energy.",
    ]
    for field in ("epsilon_range", "l2_range", "reported_linf_range"):
        groups = results["residual"]["subgroups"].get("same_image_same_" + field, {})
        count = sum(not m["small_group"] for m in groups.values())
        lines.append(
            f"- {field}: {count} comparable same-image bins with >=20 regions of each class; "
            + ("all absent or too small." if count == 0 else "counts and metrics above.")
        )
    if oracle:
        lines += [
            "",
            "## Oracle diagnostics (not deployable classification)",
            "",
            "True classes are taken from the attack annotation and propagated only to interventions on the same image. Clean and decoy-only images have no oracle class. Restoration denominators count interventions whose source prediction was wrong.",
            "",
            "| Scale | Kind | N | Known class | Initially wrong | Restoration rate | Mean true-margin gain | Mean area |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for scale, summary in oracle["validation"].items():
            for kind, m in summary["by_kind"].items():
                lines.append(
                    f"| {scale} | {kind} | {m['n']} | {m['known_true_class']} | {m['initially_wrong']} | {fmt(m['restoration_rate'])} | {fmt(m['mean_oracle_margin_gain'])} | {m['mean_area']:.1f} |"
                )
        lines += [
            "",
            "Within-image oracle comparisons; area-matched means attack/comparison area ratio in [0.8, 1.25]. This reduces, but does not remove, area and content confounding.",
            "",
            "| Scale | Comparison | Pairs | Images | Attack has larger gain | Mean gain difference |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for scale, summary in oracle["validation"].items():
            for label, m in summary["within_image"].items():
                lines.append(
                    f"| {scale} | {label} | {m['pairs']} | {m['images']} | {fmt(m['fraction_attack_larger_gain'])} | {fmt(m['mean_gain_difference'])} |"
                )
        config = json.loads((output / "resnet_config.json").read_text())
        lines += [
            "",
            f"ResNet inference runtime: {config['runtime_seconds']:.1f}s. Label-free classifier training gate: {config['label_free_gate']}.",
            "The label-free model uses original-prediction margin drop, original-class probability change and probability-vector L2 distance at both scales, plus the unchanged residual features. The original predicted class stays fixed. No true-class columns enter a classifier.",
        ]
    lines += [
        "",
        "## Representative failures",
        "",
        "For each model, the highest-confidence mistakes are shown without selecting or tuning another model.",
        "",
        "| Model | Image | Region | Truth | Attack probability |",
        "|---|---|---:|---|---:|",
    ]
    failures = []
    reference_keys = None
    for name in results:
        prediction = pd.read_csv(output / f"{name}_predictions.csv", dtype={"id": str})
        keys = list(zip(prediction.id, prediction.region_number))
        if reference_keys is None:
            reference_keys = keys
        assert keys == reference_keys, "Models evaluated on different samples/order"
        prediction["error_confidence"] = np.where(
            prediction.kind == "adversarial", 1 - prediction.probability, prediction.probability
        )
        bad = prediction[prediction.error_confidence > 0.5].nlargest(3, "error_confidence")
        for row in bad.itertuples():
            lines.append(
                f"| {name} | {row.id} | {row.region_number} | {row.kind} | {row.probability:.4f} |"
            )
        failures.append(bad.assign(model=name))
    pd.concat(failures).to_csv(output / "representative_failures.csv", index=False)
    # Include one failure from each model before filling the remaining gallery slots.
    failure_ids = pd.concat([g.head(1) for g in failures] + failures).id.drop_duplicates().head(6)
    gallery = []
    for ident in failure_ids:
        image = read_image(args.data_dir, ident)
        records = [
            dict(kind=r.kind, box=(r.x_min, r.y_min, r.x_max, r.y_max))
            for r in df[df.id == ident].itertuples()
        ]
        gallery.append((ident, image, residual_maps(image), records))
    plots(df, gallery, output)
    lines += [
        "",
        "![Representative failure images and shared-scale residual maps](experiments/gallery.png)",
        "",
        "## Interpretation and next step",
        "",
    ]
    base = results["residual"]["overall"]
    for name in ("boosting", "cnn", "resnet_label_free"):
        if name in results:
            m = results[name]["overall"]
            ranking_delta = (
                (m["strict_ranking_accuracy"] - base["strict_ranking_accuracy"])
                if m["strict_ranking_accuracy"] is not None
                and base["strict_ranking_accuracy"] is not None
                else None
            )
            lines.append(
                f"- {name} versus residual logistic: AUC difference {m['roc_auc'] - base['roc_auc']:+.4f}; ranking difference {fmt(ranking_delta)}."
            )
    if "resnet_label_free" in results and all(
        results["resnet_label_free"]["overall"][key] > base[key]
        for key in ("roc_auc", "average_precision")
    ):
        lines += [
            "",
            "Label-free frozen-ResNet sensitivity improves discrimination in this run. This supports including information from the attacked network, rather than relying solely on image residual statistics. Oracle restoration is corroborating diagnostic evidence only.",
        ]
    lines += [
        "",
        "The comparison is descriptive from a single fixed holdout. It does not establish statistical certainty. These fixed configurations do not exhaust what nonlinear classifiers or spatial models can learn. Frozen-network interventions reveal sensitivity to the attacked model, but denoising also removes natural texture.",
        "Next: quantify uncertainty by resampling held-out image IDs, review same-image comparable-strength errors, and confirm the strongest approach on independent data before attempting localization. Do not tune against this holdout or interpret these results as localization/competition performance.",
        "",
        "Per-experiment configurations, runtimes, subgroup counts, random references and predictions are in experiments/. Original baseline regions, metrics and plots remain unchanged. No test images were used.",
    ]
    report = args.output_dir / "report.md"
    if report.exists() and not (args.output_dir / "report_previous.md").exists():
        shutil.copy2(report, args.output_dir / "report_previous.md")
    report.write_text("\n".join(lines) + "\n")
    write_json(
        output / "comparison.json", {name: result["overall"] for name, result in results.items()}
    )


def write_report(args, root, selection, annotations, results):
    def fmt(x):
        return "unavailable" if x is None else f"{x:.4f}"

    counts = {
        phase: {
            kind: sum(annotations[i]["image_kind"] == kind for i in ids)
            for kind in ("ADV", "DECOY", "CLEAN")
        }
        for phase, ids in selection.items()
    }
    audit = json.loads((root / "reference/audit.json").read_text())
    lines = [
        "# From known boxes to localization",
        "",
        "## Supplied reference measurements",
        "",
        "The following table is the supplied prior reference, not an independently reproduced measurement in this milestone.",
        "",
        "| Model | AUC | AP | Annotated-box ranking |",
        "|---|---:|---:|---:|",
        "| Residual logistic | 0.675 | 0.580 | 60.7% |",
        "| Boosting | 0.666 | 0.628 | 63.2% |",
        "| Small CNN | 0.690 | 0.599 | 62.4% |",
        "| Residual + label-free ResNet | 0.969 | 0.970 | 96.3% |",
        "",
        "## Reference recovery and splits",
        "",
        audit["status"],
        f"Maximum archived-prediction discrepancy after feature replay/refit: {audit['max_probability_difference_from_archived']:.8g}. The recovered checkpoint is preserved and reused, not presented as the missing original checkpoint.",
        "The original 4000/1000 split is retained. Within the original training partition, 3200 images fit calibration-compatible scorers and 800 are reserved for calibration. The recovered frozen reference saw all 4000 and is diagnostic only; its calibration scores never choose thresholds or deployment settings.",
        f"Whole-image selection seed: {args.seed}. Candidate budget: {args.candidate_budget}. Stratified subsets approximate 80/10/10 ADV/DECOY/CLEAN; tiny smoke runs include every group. Full phase assignments are saved.",
        "",
        "| Phase | Attacked | Decoy-only | Clean |",
        "|---|---:|---:|---:|",
    ]
    for phase, c in counts.items():
        lines.append(f"| {phase} | {c['ADV']} | {c['DECOY']} | {c['CLEAN']} |")
    lines += [
        "",
        "All 50 predictors are explicitly allowlisted: 44 unchanged residual features and six label-free interventions. Candidate coordinates select pixels but are not numerical model inputs. Fields prefixed eval_ and candidate origins are evaluation-only. No oracle true-class feature is computed in this milestone.",
        "",
        "## Runs and cost",
        "",
        "| Experiment | Seconds | Original forward images | Modified forward images | Forward batches | Cached candidates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        cost = m["forward_counts"]
        lines.append(
            f"| {name} | {m['runtime_seconds']:.1f} | {cost['original_forward_images']} | {cost['intervention_forward_images']} | {cost['forward_batches']} | {cost['cached_candidates']} |"
        )
    lines += [
        "",
        "Original logits and full-image smoothed arrays are cached by image content. Candidate features are cached by exact coordinates. Every neural intervention uses the complete 64x64 image; no resizing or extra normalization is applied. The original predicted class stays fixed when computing margin/probability changes.",
    ]
    for mode in ("robustness", "background", "ablation", "proposals"):
        if mode not in results:
            lines += ["", f"{mode}: not run in this output directory."]
    if "robustness" in results:
        r = results["robustness"]
        lines += [
            "",
            "## Imperfect boxes: annotation-seeded diagnostics",
            "",
            "Each source attack/decoy produces at most the fixed variant budget using offsets 0, +/-2, +/-4, +/-8 and scales 0.75/1/1.25. Clamping and global deduplication can reduce counts. All exact annotations are excluded from perturbed-only pools. Targets are recomputed IoUs, never inherited source labels.",
            "",
            "| Scorer | Pool | Candidates | Images | Binary AUC | AP | Positive prevalence | Best IoU | Selected IoU | Spearman |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, pools in r["pools"].items():
            for pool, phases in pools.items():
                m = phases["holdout"]
                if "candidates" in m:
                    lines.append(
                        f"| {name} | {pool} | {m['candidates']} | {m['images']} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | {fmt(m['attack_prevalence'])} | {fmt(m['mean_best_iou'])} | {fmt(m['mean_selected_iou'])} | {fmt(m['iou_score_spearman'])} |"
                    )
        lines += [
            "",
            "Binary positives: IoU>=0.5; negatives: IoU<=0.1. Ambiguous candidates are excluded only from binary metrics, not continuous-IoU, correlation, selection or coverage metrics. These are experimental conventions, not the official image score.",
            f"Variant adaptation run: {r['adaptation']['run']}. Reason: {r['adaptation']['reason']}. Training counts and configuration are recorded in robustness_metrics.json.",
        ]
        m = r["pools"]["frozen_reference"]["perturbed_only"]["holdout"]
        lines += [
            "",
            f"Perturbed-only useful-box recall at IoU 0.3/0.5/0.7: {m['useful_box_recall']}.",
            "",
            "Frozen-reference score distribution on perturbed holdout candidates:",
            "",
            "| Actual IoU bin | Candidates | Mean score | Q25 | Median | Q75 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for label, bin_metrics in m["iou_bins"].items():
            q = bin_metrics["score_quantiles"]
            lines.append(
                f"| {label} | {bin_metrics['n']} | {fmt(bin_metrics['score_mean'])} | {fmt(q[1])} | {fmt(q[2])} | {fmt(q[3])} |"
            )
    if "background" in results:
        r = results["background"]
        lines += [
            "",
            "## Background and no-attack diagnostics",
            "",
            "Controls combine ordinary and high-texture grid regions avoiding all marked boxes. Mixed diagnostic pools prefix exact marked regions then controls, at fixed total budgets. These annotation-seeded pools are not image-only localization. Control-only FPRs and full maximum-score quantiles are in background_metrics.json.",
            "",
            "| Budget | Scorer | Calibrated threshold | Candidate-negative FPR | Clean image FPR | Decoy-only image FPR | Diagnostic image score |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for budget, values in r["budgets"].items():
            for name, record in values["calibrated"].items():
                m = record["holdout"]
                lines.append(
                    f"| {budget} | {name} | {fmt(record['calibration']['threshold'])} | {fmt(m['candidate_negative_fpr'])} | {fmt(m['groups'].get('CLEAN', {}).get('any_candidate_exceeds_threshold'))} | {fmt(m['groups'].get('DECOY', {}).get('any_candidate_exceeds_threshold'))} | {fmt(m['official_mean'])} |"
                )
        lines += [
            "",
            f"Background adaptation retained {r['adaptation']['known_decoy_negatives']} annotated decoy negatives and added {r['adaptation']['added_negative_regions']} background negatives. Attack-vs-background results and the original attack-vs-decoy benchmark are separate.",
        ]
        lines += [
            "",
            "Candidate FPR versus image-max FPR, at each calibrated threshold:",
            "",
            "| Budget | Scorer | No-attack group | Images | Negative candidates | Candidate FPR | Image FPR |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for budget, values in r["budgets"].items():
            for name, record in values["calibrated"].items():
                for kind, g in record["holdout"]["groups"].items():
                    if kind != "ADV" and "negative_candidates" in g:
                        lines.append(
                            f"| {budget} | {name} | {kind} | {g['images']} | {g['negative_candidates']} | {fmt(g['candidate_negative_fpr'])} | {fmt(g['any_candidate_exceeds_threshold'])} |"
                        )
    if "ablation" in results:
        r = results["ablation"]
        lines += [
            "",
            "## Known-box feature ablation",
            "",
            "Archived feature replay on identical annotated regions; all three models use the same original training samples and logistic configuration. This does not re-extract neural features or independently reproduce the supplied reference end to end.",
            "",
            "| Scorer | Regions | AUC | AP | AP prevalence reference | Ranking | Random ranking reference |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name in ("residual_reference", "resnet_reference", "frozen_reference"):
            m = r[name]["overall"]
            lines.append(
                f"| {name} | {m['n']} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | {fmt(m['random_average_precision'])} | {fmt(m['strict_ranking_accuracy'])} | {fmt(m['random_ranking_accuracy'])} |"
            )
        lines += [
            "",
            f"Combined minus ResNet-only: {r['combined_minus_resnet_only']}. Residual-only requires no neural forwards; ResNet-only and combined require the same two modified-image forwards per region plus one cached original per image.",
        ]
    if "proposals" in results:
        r = results["proposals"]
        lines += [
            "",
            "## Image-only proposals",
            "",
            f"Fixed published-prior size bank {SIZES}, stride 8, all boundaries included. {r['grid_candidates_per_image']} grid rectangles per image; at most {r['scored_budget']} receive ResNet scoring. The shortlist uses a training-only residual model and fixed geometric diversity. No ground-truth or decoy coordinates seed these proposals.",
            "",
            "| Pool | Attacked images | Best available IoU | Recall @0.3 | Recall @0.5 | Recall @0.7 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, key in [
            ("Full grid", "before_shortlisting"),
            ("Shortlist", "after_shortlisting"),
        ]:
            m = r[key]["holdout"]
            lines.append(
                f"| {name} | {m['attacked_images']} | {fmt(m['mean_best_iou'])} | {fmt(m['recall']['0.3'])} | {fmt(m['recall']['0.5'])} | {fmt(m['recall']['0.7'])} |"
            )
        lines += [
            "",
            "| Scorer | Selected IoU before rejection | Gap to shortlist best | Calibration image score | Holdout image score |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, phases in r["candidate_models"].items():
            m = phases["holdout"]
            calibrated = r["calibrated"].get(name, {})
            lines.append(
                f"| {name} | {fmt(m['mean_selected_iou'])} | {fmt(m['selection_gap'])} | {fmt(calibrated.get('calibration', {}).get('official_mean'))} | {fmt(calibrated.get('holdout', {}).get('official_mean'))} |"
            )
        lines += [
            "",
            f"Selected using calibration only: {r['selected_pipeline']['scorer']}, rejection threshold {r['selected_pipeline']['threshold']:.6g}.",
            f"Final holdout score: {r['final_holdout']['official_mean']:.4f}; all-zero baseline on this population: {r['final_holdout']['all_zero_baseline']:.4f}.",
            "",
            "| Group | Images | Mean official score | Abstention rate |",
            "|---|---:|---:|---:|",
        ]
        for kind, m in r["final_holdout"]["groups"].items():
            lines.append(
                f"| {kind} | {m['images']} | {m['official_mean']:.4f} | {m['abstention_rate']:.4f} |"
            )
        lines += [
            "",
            "Maximum candidate-score distributions for the selected pipeline:",
            "",
            "| Group | Min | Q25 | Median | Q75 | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for kind, m in r["final_holdout"]["groups"].items():
            lines.append(
                "| " + kind + " | " + " | ".join(fmt(v) for v in m["max_score_quantiles"]) + " |"
            )
        reg = r["quality_regression"]
        lines += [
            "",
            f"Quality regression run: {reg['run']}; calibration median selected/true area ratio: {fmt(reg['calibration_median_selected_area_ratio'])}. "
            f"Trigger: {reg['trigger']}. Training images: {reg.get('training_images', 0)}; candidates: {reg.get('training_candidates', 0)}. "
            "The fixed configuration and explicit feature list are saved in proposals_metrics.json.",
            "![Holdout failures](proposal_failures.png)",
            "",
            "The official score here is mean IoU for attacked images (zero on abstention), and one only for an all-zero output on no-attack images. Exactly one integer box or all-zero box is returned. Frozen-reference scores remain diagnostic; final thresholds and scorer choice use only reserved calibration images.",
            "",
            "Full-grid and shortlist coverage are geometric limits for a selector returning those rectangles unchanged. A high known-box ranking or a large original-class margin drop does not imply accurate boundaries.",
        ]
    lines += [
        "",
        "## Subgroups and original benchmark",
        "",
        "The JSON metrics include counts by attack family, source-image attack norm/epsilon bins, JPEG quality and source-image decoy-type composition. Robustness groups use all relevant candidates from the same source-image subset. Original annotated-box ablation uses the earlier paired decoy-type definitions. Missing norms are not zero, and nominal strength is not realized residual energy. Small subsets support descriptive comparisons only.",
    ]
    for mode in ("robustness", "background"):
        if mode in results:
            lines += [
                "",
                f"### {mode}: attack-versus-decoy retained separately",
                "",
                "| Scorer | Attacks | Decoys | AUC | AP | Ranking |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for name, record in results[mode]["original_benchmark"].items():
                m = record["overall"]
                lines.append(
                    f"| {name} | {m['adversarial']} | {m['decoy']} | {fmt(m['roc_auc'])} | {fmt(m['average_precision'])} | {fmt(m['strict_ranking_accuracy'])} |"
                )
    if "proposals" in results:
        for key in (
            "attack_family",
            "jpeg_quality",
            "eval_attack_epsilon",
            "eval_attack_l2",
            "eval_attack_reported_linf",
            "eval_decoy_types",
        ):
            lines += [
                "",
                f"### Proposal holdout: {key}",
                "",
                "| Group | Images | Candidates | Binary positives | Binary negatives | AUC | Best IoU | Selected IoU |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for label, m in results["proposals"]["holdout_subgroups"].get(key, {}).items():
                lines.append(
                    f"| {label} | {m['images']} | {m['candidates']} | {m['positives']} | {m['negatives']} | {fmt(m['roc_auc'])} | {fmt(m['mean_best_iou'])} | {fmt(m['mean_selected_iou'])} |"
                )
    lines += [
        "",
        "## Limits and next action",
        "",
        "Candidate-heavy experiments are limited to the explicitly saved whole-image subset when --limit is used. This is not a full-dataset localization result. Model fitting still uses the original cached training regions; calibration and final evaluation use separate image IDs. Background adaptations do not redefine the original attack-vs-decoy benchmark.",
        "If shortlisting loses useful boxes, improve proposal coverage or shortlist diversity before interpreting classifier performance. If selecting among good proposals is the main gap, validate the candidate-quality regression on a larger calibration population. Expensive exhaustive scoring motivates a small attack+decoy union-mask segmenter as a future proposal stage; no localization network is trained here.",
        "Local boundary refinement, a full-grid neural sweep, and a learned localizer were not run. No new attacks, test-image fitting, leaderboard feedback or competition submissions were used. No statistical certainty is claimed from a single small holdout.",
    ]
    if "proposals" in results:
        p = results["proposals"]
        full, short = p["before_shortlisting"]["holdout"], p["after_shortlisting"]["holdout"]
        lost = full["recall"]["0.5"] - short["recall"]["0.5"]
        lines += [
            "",
            f"Measured shortlist loss at IoU 0.5: {lost:.1%} of attacked holdout images. "
            f"Mean best IoU fell from {full['mean_best_iou']:.4f} to {short['mean_best_iou']:.4f}.",
        ]
        if lost > 0.1:
            lines += [
                "Next action justified by this run: improve the cheap proposal shortlist on training/calibration images before increasing scorer complexity. Compare a larger fixed shortlist or a small union-of-attack-and-decoy mask segmenter; measure recall and cost before another final evaluation."
            ]
        else:
            lines += [
                "Next action justified by this run: focus on selecting accurate boundaries among available candidates, validating quality regression or a small fixed boundary-refinement budget on training/calibration images before another final evaluation."
            ]
        n = sum(map(len, selection.values()))
        lines += [
            "",
            f"Runtime planning only: proposals took {p['runtime_seconds']:.1f}s for {n} images "
            f"({p['runtime_seconds'] / n:.2f}s/image, including residual grid generation, cached-feature reuse and evaluation). "
            f"A linear projection to 5000 images at the same budget is {p['runtime_seconds'] / n * 5000 / 3600:.2f} hours; "
            "this is not a measured full run and depends on cache/hardware. "
            f"Uncached neural work for the full grid would be {2 * p['grid_candidates_per_image']} modified-image forwards per image, "
            f"versus {2 * p['scored_budget']} at the tested shortlist budget.",
        ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
