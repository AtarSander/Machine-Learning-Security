"""Metrics for patch analysis experiments."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import COORDS as base_COORDS
from src.data import SCALES, output_box


def score_metrics(frame, prediction):
    y = (frame.kind == "adversarial").astype(int)
    result = dict(
        n=len(frame),
        images=int(frame.id.nunique()),
        adversarial=int(y.sum()),
        decoy=int((1 - y).sum()),
    )
    result["confusion_matrix"] = (
        confusion_matrix(y, frame[prediction] >= 0.5, labels=[0, 1]).tolist()
        if len(frame)
        else [[0, 0], [0, 0]]
    )
    result["roc_auc"] = float(roc_auc_score(y, frame[prediction])) if y.nunique() == 2 else None
    result["average_precision"] = (
        float(average_precision_score(y, frame[prediction])) if y.nunique() == 2 else None
    )
    wins, random_choices = [], []
    for _, g in frame.groupby("id"):
        if set(g.kind) == {"adversarial", "decoy"}:
            wins.append(
                float(
                    g.loc[g.kind == "adversarial", prediction].min()
                    > g.loc[g.kind == "decoy", prediction].max()
                )
            )
            random_choices.append(1 / (1 + int((g.kind == "decoy").sum())))
    result.update(
        ranking_images=len(wins), strict_ranking_accuracy=float(np.mean(wins)) if wins else None
    )
    result["small_group"] = min(result["adversarial"], result["decoy"]) < 20
    result["random_average_precision"] = float(y.mean()) if len(y) else None
    result["random_ranking_accuracy"] = float(np.mean(random_choices)) if random_choices else None
    return result


def evaluate(df, image_ids, seed, manifest=None):
    df["partition"] = "unavailable"
    if len(image_ids) < 2:
        return {"unavailable": "fewer than two images"}
    if manifest is None:
        train, val = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                image_ids, groups=image_ids
            )
        )
        train_ids, val_ids = set(np.array(image_ids)[train]), set(np.array(image_ids)[val])
    else:
        train_ids = set(manifest["train_image_ids"]) & set(image_ids)
        val_ids = set(manifest["validation_image_ids"]) & set(image_ids)
        if train_ids | val_ids != set(image_ids):
            raise ValueError("Persisted split does not cover the selected images")
    assert not train_ids & val_ids
    df["partition"] = np.where(df.id.isin(train_ids), "train", "validation")
    binary = df.kind != "background"
    tr, va = df[binary & (df.partition == "train")], df[binary & (df.partition == "validation")]
    result = dict(train_image_ids=sorted(train_ids), validation_image_ids=sorted(val_ids))
    if tr.kind.nunique() != 2 or va.kind.nunique() != 2:
        result["unavailable"] = "both classes required in each partition"
        return result
    all_features = [c for c in df if c.startswith(("s0.7_", "s1.2_"))]
    for name, columns in [
        ("energy", [f"s{s}_rms_all" for s in SCALES]),
        ("residual", all_features),
    ]:
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=seed)
        )
        model.fit(tr[columns], (tr.kind == "adversarial").astype(int))
        pred = name + "_probability"
        df[pred] = np.nan
        df.loc[va.index, pred] = model.predict_proba(va[columns])[:, 1]
        scored = df.loc[va.index]
        detail = dict(
            features=columns,
            overall=score_metrics(scored, pred),
            breakdowns={},
            standardized_coefficients=dict(zip(columns, model[-1].coef_[0].tolist())),
        )
        for key in (
            "attack_family",
            "jpeg_quality",
            "epsilon_range",
            "l2_range",
            "reported_linf_range",
        ):
            detail["breakdowns"][key] = {
                str(k): score_metrics(g, pred) for k, g in scored.groupby(key, dropna=False)
            }
        result[name] = detail
    return result


def subgroup_frames(df):
    # Strength subgroups use the attack's metadata to select entire source images.
    for key in ("attack_family", "jpeg_quality"):
        for label, g in df.groupby(key, dropna=False):
            yield key, str(label), g
    attacks = df[df.kind == "adversarial"]
    for key in ("epsilon_range", "l2_range", "reported_linf_range"):
        for label, g in attacks.groupby(key, dropna=False):
            yield "source_attack_" + key, str(label), df[df.id.isin(g.id)]
        for label, g in attacks.groupby(key, dropna=False):
            if pd.isna(label) or str(label) == "missing":
                continue
            decoys = df[(df.kind == "decoy") & df.id.isin(g.id) & (df[key] == label)]
            matched = pd.concat([g[g.id.isin(decoys.id)], decoys]).sort_index()
            if len(matched):
                yield "same_image_same_" + key, str(label), matched
    for decoy_type, decoys in df[df.kind == "decoy"].groupby("region_type"):
        # Exclude decoy-only images for this paired type comparison.
        decoys = decoys[decoys.id.isin(attacks.id)]
        yield (
            "paired_decoy_type",
            str(decoy_type),
            pd.concat([attacks[attacks.id.isin(decoys.id)], decoys]).sort_index(),
        )


def experiment_metrics(df, prediction):
    result = dict(overall=score_metrics(df, prediction), subgroups={})
    for key, label, group in subgroup_frames(df):
        result["subgroups"].setdefault(key, {})[label] = score_metrics(group, prediction)
    return result


def selected_rows(table, score):
    if table.empty:
        return table.copy()
    return table.loc[table.groupby("id", sort=False)[score].idxmax()].copy()


def pipeline_metrics(table, score, threshold):
    selected = selected_rows(table, score)
    selected["abstained"] = selected[score] < threshold
    selected["eval_official_score"] = np.where(
        selected.abstained,
        ~selected.eval_has_attack,
        np.where(selected.eval_has_attack, selected.eval_iou, 0),
    ).astype(float)
    for c in base_COORDS:
        selected["output_" + c] = np.where(selected.abstained, 0, selected[c]).astype(int)
    for box in selected[["output_" + c for c in base_COORDS]].itertuples(index=False, name=None):
        output_box(box)
    summary = dict(
        images=len(selected),
        threshold=float(threshold),
        official_mean=float(selected.eval_official_score.mean()) if len(selected) else None,
        all_zero_baseline=float((~selected.eval_has_attack).mean()) if len(selected) else None,
        groups={},
    )
    for kind, g in selected.groupby("image_kind"):
        negative_candidates = table[(table.image_kind == kind) & (table.eval_iou <= 0.1)]
        summary["groups"][kind] = dict(
            images=len(g),
            official_mean=float(g.eval_official_score.mean()),
            abstention_rate=float(g.abstained.mean()),
            max_score_quantiles=g[score].quantile([0, 0.25, 0.5, 0.75, 1]).tolist(),
            any_candidate_exceeds_threshold=float((~g.abstained).mean()),
            negative_candidates=len(negative_candidates),
            candidate_negative_fpr=float((negative_candidates[score] >= threshold).mean())
            if len(negative_candidates)
            else None,
        )
    return summary, selected


def calibrate(table, score):
    if table.empty or set(table.phase) != {"calibration"}:
        raise ValueError("Threshold calibration requires reserved calibration images")
    selected = selected_rows(table, score)
    values = selected[score].to_numpy()
    thresholds = sorted(set([0.0] + values.tolist() + [float(np.nextafter(values.max(), np.inf))]))
    curve = []
    for threshold in thresholds:
        scores = np.where(
            values >= threshold,
            np.where(selected.eval_has_attack, selected.eval_iou, 0),
            ~selected.eval_has_attack,
        )
        curve.append(dict(threshold=threshold, official_mean=float(scores.mean())))
    best = max(curve, key=lambda r: (r["official_mean"], r["threshold"]))
    return best, curve


def pool_metrics(table, score):
    if table.empty:
        return {"unavailable": "No candidates in this pool"}
    binary = table[(table.eval_iou >= 0.5) | (table.eval_iou <= 0.1)]
    y = binary.eval_iou >= 0.5
    result = dict(
        candidates=len(table),
        images=int(table.id.nunique()),
        ambiguous_excluded=len(table) - len(binary),
        binary_candidates=len(binary),
        positives=int(y.sum()),
        negatives=int((~y).sum()),
        roc_auc=float(roc_auc_score(y, binary[score])) if y.nunique() == 2 else None,
        average_precision=float(average_precision_score(y, binary[score]))
        if y.nunique() == 2
        else None,
        attack_prevalence=float(y.mean()) if len(y) else None,
        negative_fpr_at_half=float((binary.loc[~y, score] >= 0.5).mean()) if (~y).any() else None,
        iou_score_spearman=float(table.eval_iou.corr(table[score], method="spearman"))
        if table.eval_iou.nunique() > 1 and table[score].nunique() > 1
        else None,
        iou_bins={},
    )
    bins = pd.cut(table.eval_iou, [-1e-9, 0.1, 0.3, 0.5, 0.7, 1.000001])
    for label, g in table.groupby(bins, observed=True):
        result["iou_bins"][str(label)] = dict(
            n=len(g),
            score_mean=float(g[score].mean()),
            score_quantiles=g[score].quantile([0, 0.25, 0.5, 0.75, 1]).tolist(),
        )
    attacked = table[table.eval_has_attack]
    best = attacked.groupby("id").eval_iou.max()
    picked = selected_rows(attacked, score)
    result["attacked_images"] = len(best)
    result["mean_best_iou"] = float(best.mean()) if len(best) else None
    result["mean_selected_iou"] = float(picked.eval_iou.mean()) if len(best) else None
    result["selection_gap"] = float(best.mean() - picked.eval_iou.mean()) if len(best) else None
    result["useful_box_recall"] = {
        str(t): float((best >= t).mean()) if len(best) else None for t in (0.3, 0.5, 0.7)
    }
    result["selected_box_recall"] = {
        str(t): float((picked.eval_iou >= t).mean()) if len(best) else None for t in (0.3, 0.5, 0.7)
    }
    result["unrejected_at_half"] = pipeline_metrics(table, score, 0.5)[0]
    return result


def subgroup_metrics(table, score):
    result = {}
    for key in ("attack_family", "jpeg_quality", "eval_decoy_types"):
        result[key] = {str(k): pool_metrics(g, score) for k, g in table.groupby(key, dropna=False)}
    for key in ("epsilon", "l2", "reported_linf"):
        col = "eval_attack_" + key
        bins = (
            [0, 0.5, 1, 2, np.inf]
            if key == "l2"
            else [0, 4 / 255, 8 / 255, 12 / 255, 20 / 255, np.inf]
        )
        labels = (
            pd.cut(pd.to_numeric(table[col], errors="coerce"), bins, include_lowest=True)
            .astype("string")
            .fillna("missing")
        )
        result[col] = {
            str(k): pool_metrics(g, score) for k, g in table.groupby(labels, dropna=False)
        }
    return result


def original_benchmark(rows, selection, models, columns):
    heldout = rows[(rows.kind != "background") & rows.id.isin(selection["holdout"])].copy()
    result = {}
    for name, model in models.items():
        if name in ("frozen_reference", "fit_combined", "adapted_robustness", "adapted_background"):
            heldout["probability"] = model.predict_proba(heldout[columns])[:, 1]
            result[name] = base_experiment_metrics(heldout, "probability")
    return result


def phase_reports(table, score):
    return {
        phase: pool_metrics(table[table.phase == phase], score)
        for phase in ("calibration", "holdout")
    }


def grid_coverage(records):
    result = {}
    for phase, group in records.groupby("phase"):
        attacked = group[group.eval_has_attack]
        best = attacked.groupby("id").eval_iou.max()
        result[phase] = dict(
            images=int(group.id.nunique()),
            candidates=len(group),
            attacked_images=len(best),
            mean_best_iou=float(best.mean()) if len(best) else None,
            recall={
                str(t): float((best >= t).mean()) if len(best) else None for t in (0.3, 0.5, 0.7)
            },
        )
    return result


base_experiment_metrics = experiment_metrics
