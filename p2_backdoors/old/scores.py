"""Score files (results/<detector>.csv: id,c00,...,c19, one row per model) and everything done across models:
per-class normalization within one population, and the submission check.

Run `uv run python scores.py` to compare raw and normalized scores of the COMPARE detectors."""
import csv
import math
import os

import torch

from data import RESULTS_DIR, SHADOW_LABELS, TRAIN_LABELS, test_model_ids
from .evaluate import clean_acc_correlation, evaluate_detector, evaluate_train, subtract_model_median
from model import N_CLASSES

CLASS_COLUMNS = [f"c{c:02d}" for c in range(N_CLASSES)]
# A class's MAD across models is raised to at least this fraction of the median MAD over the 20 classes.
# Without a floor, a class that never varies (runner_up has one with MAD 0) divides by zero, and the portal
# rejects nan/inf. At 0.5 the floor is one image's worth of runner_up (1/190) and leaves every cos_other
# class untouched (AGENTS.md finding 25).
MAD_FLOOR = 0.5
COMPARE = ["cos_other", "cos_source", "runner_up", "own_margin"]  # main(): raw vs normalized, shadows and train
SUBMIT = "cos_other"  # main(): its test scores, normalized among the test models, become a submission file


def read_scores(path):
    """{model_id: N_CLASSES floats} from a score file."""
    with open(path) as f:
        return {row["id"]: [float(row[column]) for column in CLASS_COLUMNS] for row in csv.DictReader(f)}


def write_scores(path, scores):
    # Write a temporary file and rename it: stopping a run mid-write then leaves the old file, never half a file.
    with open(path + ".tmp", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + CLASS_COLUMNS)
        for model_id, row in scores.items():
            # csv writes floats with str(), which reads back as exactly the same number: no rounding ties.
            writer.writerow([model_id] + row)
    os.replace(path + ".tmp", path)


def check_submission(path, test_ids):
    """Crash unless the file passes the portal's checks: a rejected file still uses up a daily submission.
    Reads the file as text, so it checks exactly what would be uploaded."""
    with open(path) as f:
        reader = csv.reader(f)
        assert next(reader) == ["id"] + CLASS_COLUMNS, "wrong header"
        ids = []
        for row in reader:
            values = [float(value) for value in row[1:]]
            assert len(values) == N_CLASSES, f"{row[0]}: {len(values)} values instead of {N_CLASSES}"
            assert all(math.isfinite(value) for value in values), f"{row[0]}: nan or inf"
            ids.append(row[0])
    assert len(ids) == len(set(ids)), "an id is repeated"
    assert set(ids) == set(test_ids), f"ids differ from the test models: {len(set(test_ids) - set(ids))} missing, " \
                                      f"{len(set(ids) - set(test_ids))} unknown"


def normalize_per_class(scores):
    """Robust z-score of each class across the models of one population: (score - median) / MAD, per class.
    A class that is naturally attractive in every model is then measured against its own usual level.
    Never mix populations: shadows are normalized with shadows, test models with test models."""
    ids = list(scores)
    table = torch.tensor([scores[model_id] for model_id in ids], dtype=torch.float64)  # (models, classes)
    median = table.quantile(0.5, dim=0)
    mad = (table - median).abs().quantile(0.5, dim=0)
    mad = mad.clamp(min=MAD_FLOOR * mad.quantile(0.5).item())
    return dict(zip(ids, ((table - median) / mad).tolist()))


def compare(name):
    """Print raw vs normalized AUC of one detector: per group on the shadows, and on the 6 train models."""
    raw = read_scores(os.path.join(RESULTS_DIR, name + ".csv"))
    norm = normalize_per_class(raw)
    columns = [evaluate_detector(s, SHADOW_LABELS) for s in (raw, norm, subtract_model_median(raw),
                                                             subtract_model_median(norm))]
    print(f"\n{name:<17} {'raw':>6} {'norm':>6} | within {'raw':>6} {'norm':>6} | pos")
    for group, (_, n_positives) in columns[0].items():
        raw_auc, norm_auc, raw_within, norm_within = (column[group][0] for column in columns)
        print(f"{group:<17} {raw_auc:6.3f} {norm_auc:6.3f} |        {raw_within:6.3f} {norm_within:6.3f} | {n_positives:3d}")
    print(f"corr(max score, clean_acc): raw {clean_acc_correlation(raw, SHADOW_LABELS):+.2f}, "
          f"norm {clean_acc_correlation(norm, SHADOW_LABELS):+.2f}")
    # The 6 train models are their own population: a median and MAD over 6 models is only a rough hint.
    train = read_scores(os.path.join(RESULTS_DIR, "train", name + ".csv"))
    raw_auc, raw_ranks = evaluate_train(train, TRAIN_LABELS)
    norm_auc, norm_ranks = evaluate_train(normalize_per_class(train), TRAIN_LABELS)
    print(f"train models (normalized among their 6): AUC {raw_auc:.3f} -> {norm_auc:.3f}, "
          f"target ranks (15, 3, 7) {[f'{r:g}' for r in raw_ranks]} -> {[f'{r:g}' for r in norm_ranks]}")


def main():
    for name in COMPARE:
        compare(name)
    raw_test = read_scores(os.path.join(RESULTS_DIR, "test", SUBMIT + ".csv"))
    path = os.path.join(RESULTS_DIR, "test", SUBMIT + "_norm.csv")
    write_scores(path, normalize_per_class(raw_test))
    check_submission(path, test_model_ids())
    print(f"\n{path}: normalized among the {len(raw_test)} test models, passes the portal's checks")


if __name__ == "__main__":
    main()
