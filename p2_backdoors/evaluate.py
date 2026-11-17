import csv

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


def evaluate_detector(scores, labels_csv):
    """AUC over all (shadow, class) pairs of the valid shadows: overall, per trigger kind and per mode.

    scores: {shadow_id: N_CLASSES numbers}, higher = more likely that class is a backdoor target.
    Returns {group: (auc, number of positives)}. Each group's positives are compared with ALL negatives.
    """
    positives = []  # (score, kind, mode) for every (shadow, target class) pair
    negatives = []  # score for every (shadow, class) pair that is not a target
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            if int(row["valid"]) == 0:
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
