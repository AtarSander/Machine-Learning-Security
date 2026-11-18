import os
import time

import torch

import detectors
from data import (
    SHADOW_DIR,
    TRAIN_DIR,
    get_device,
    load_clean_images,
    load_shadow_model,
    load_test_model,
    load_train_model,
    split_ab,
    test_model_ids,
)
from evaluate import auc, clean_acc_correlation, evaluate_detector, parse_list, read_labels, subtract_model_median
from model import N_CLASSES
from scores import check_submission, read_scores, write_scores

DETECTOR = "neg_clean_acc"  # a key of DETECTORS
# "shadows": all valid shadows, "dev": DEV_SHADOWS only, "train": the 6 organizer train models,
# "test": the 200 test models (the CSV is then a submission, checked against the portal's rules)
RUN = "shadows"
DETECTORS = {
    "random": detectors.random_scores,
    "neg_clean_acc": detectors.neg_clean_acc,
    "cos_other": detectors.cos_other,
    "cos_source": detectors.cos_source,
    "runner_up": detectors.runner_up,
    "own_margin": detectors.own_margin,
}
# For developing expensive detectors. Drawn once (Phase 1 Step 1): torch.randperm with Generator seed 0,
# 8 of the 47 clean shadows, then 8 of the 17 valid poisoned ones (12 positives, every kind and mode).
DEV_SHADOWS = [
    "shadow003", "shadow022", "shadow046", "shadow050", "shadow056", "shadow058", "shadow066", "shadow069",
    "shadow001", "shadow007", "shadow016", "shadow024", "shadow043", "shadow045", "shadow051", "shadow072",
]
RESULTS_DIR = "results"
SHADOW_LABELS = os.path.join(SHADOW_DIR, "labels.csv")
TRAIN_LABELS = os.path.join(TRAIN_DIR, "labels.csv")


def report_shadows(scores, rows):
    n_poisoned = sum(1 for row in rows if row["id"] in scores and row["targets"])
    print(f"\n{DETECTOR}: {len(scores)} shadows scored ({len(scores) - n_poisoned} clean, {n_poisoned} poisoned)")
    results = evaluate_detector(scores, SHADOW_LABELS)
    within = evaluate_detector(subtract_model_median(scores), SHADOW_LABELS)
    print(f"{'group':<17} {'AUC':>6} {'within':>7} {'pos':>4}")
    for group, (value, n_positives) in results.items():
        print(f"{group:<17} {value:6.3f} {within[group][0]:7.3f} {n_positives:4d}")
    print(f"correlation of a model's max score with its clean_acc: {clean_acc_correlation(scores, SHADOW_LABELS):+.2f}")


def report_train(scores, rows):
    """Where the true target ranks among the classes of each poisoned train model (1 = highest score)."""
    print(f"\n{DETECTOR} on the organizer train models")
    positives = []
    negatives = []
    for row in rows:
        model_scores = scores[row["id"]]
        targets = [int(t) for t in parse_list(row["targets"])]
        line = f"{row['id']} max {max(model_scores):+.4f}"
        for t in targets:
            # Ties count as half, as in the AUC: with 20 equal scores the target is at rank 10.5.
            higher = sum(s > model_scores[t] for s in model_scores)
            equal = sum(s == model_scores[t] for s in model_scores) - 1
            line += f" | target {t} rank {1 + higher + 0.5 * equal:g}/{N_CLASSES}"
        print(line)
        for c in range(N_CLASSES):
            if c in targets:
                positives.append(model_scores[c])
            else:
                negatives.append(model_scores[c])
    print(f"AUC {auc(positives, negatives):.3f} ({len(positives)} positives)")


def main():
    device = get_device()
    images, labels = load_clean_images()
    _, b = split_ab(labels)
    # Only B: the shadows were finetuned on A. The train models get B too, so every run sees the same images.
    images_b, labels_b = images[b].to(device), labels[b].to(device)

    if RUN == "train":
        rows = read_labels(TRAIN_LABELS)
        model_ids = [row["id"] for row in rows]
        load_model = load_train_model
    elif RUN == "test":
        rows = [{"id": model_id, "targets": "?"} for model_id in test_model_ids()]  # the test labels are secret
        model_ids = [row["id"] for row in rows]
        load_model = load_test_model
    else:
        rows = read_labels(SHADOW_LABELS)
        model_ids = DEV_SHADOWS if RUN == "dev" else [row["id"] for row in rows if row["valid"] == "1"]
        load_model = load_shadow_model
    targets = {row["id"]: row["targets"] for row in rows}
    results_dir = RESULTS_DIR if RUN == "shadows" else os.path.join(RESULTS_DIR, RUN)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, DETECTOR + ".csv")

    # Resumable: models already in the file are skipped, and the file is rewritten after every model.
    scores = read_scores(path) if os.path.exists(path) else {}
    torch.manual_seed(0)  # only the random baseline uses the global RNG
    for model_id in model_ids:
        if model_id in scores:
            continue
        start_time = time.time()
        net = load_model(model_id, device)
        scores[model_id] = DETECTORS[DETECTOR](net, images_b, labels_b).cpu().tolist()
        write_scores(path, scores)
        top3 = torch.tensor(scores[model_id]).topk(3).indices.tolist()
        print(f"{model_id} top3 {top3} targets [{targets[model_id]}] | {time.time() - start_time:.1f}s")

    if RUN == "train":
        report_train(scores, rows)
    elif RUN == "test":
        check_submission(path, model_ids)
        print(f"\n{path}: {len(scores)} test models, passes the portal's checks")
    else:
        report_shadows(scores, rows)


if __name__ == "__main__":
    main()
