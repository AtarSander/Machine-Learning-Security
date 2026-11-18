import csv
import statistics

from model import N_CLASSES


def auc(pos, neg):
    """Chance that a random positive scores higher than a random negative; a tie counts as half."""
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def parse_list(text):
    # Clean shadows have empty columns, and "".split(";") would give [""] instead of [].
    return text.split(";") if text else []


def read_labels(labels_csv):
    with open(labels_csv) as f:
        return list(csv.DictReader(f))


def evaluate_detector(scores, labels_csv):
    """AUC over all (shadow, class) pairs of the valid shadows in `scores`: overall, per trigger kind and per mode.

    scores: {shadow_id: N_CLASSES numbers}, higher = more likely that class is a backdoor target.
    Shadows missing from `scores` are skipped, so a dev-subset or unfinished run can be evaluated.
    Returns {group: (auc, number of positives)}. Each group's positives are compared with ALL negatives.
    """
    positives = []  # (score, kind, mode) for every (shadow, target class) pair
    negatives = []  # score for every (shadow, class) pair that is not a target
    for row in read_labels(labels_csv):
        if int(row["valid"]) == 0 or row["id"] not in scores:
            continue
        targets = [int(t) for t in parse_list(row["targets"])]
        kinds = parse_list(row["kinds"])
        modes = parse_list(row["modes"])
        for c in range(N_CLASSES):
            score = scores[row["id"]][c]
            if c in targets:
                i = targets.index(c)
                positives.append((score, kinds[i], modes[i]))
            else:
                negatives.append(score)

    results = {"all": (auc([s for s, _, _ in positives], negatives), len(positives))}
    for kind in sorted({k for _, k, _ in positives}):
        group = [s for s, k, _ in positives if k == kind]
        results["kind=" + kind] = (auc(group, negatives), len(group))
    for mode in sorted({m for _, _, m in positives}):
        group = [s for s, _, m in positives if m == mode]
        results["mode=" + mode] = (auc(group, negatives), len(group))
    return results


def subtract_model_median(scores):
    """Every model's scores minus that model's median score. This removes anything shared by all classes of a
    model, such as a dependence on clean accuracy, so the AUC that survives comes from ranking the classes."""
    centered = {}
    for model_id, row in scores.items():
        median = statistics.median(row)
        centered[model_id] = [s - median for s in row]
    return centered


def clean_acc_correlation(scores, labels_csv):
    """Pearson correlation, over the shadows in `scores`, between a model's highest score and its clean accuracy."""
    rows = [row for row in read_labels(labels_csv) if row["id"] in scores]
    max_scores = [max(scores[row["id"]]) for row in rows]
    clean_accs = [float(row["clean_acc"]) for row in rows]
    return statistics.correlation(max_scores, clean_accs)
