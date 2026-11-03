"""Noise pipeline for patch analysis experiments."""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.candidates import CandidateScorer as loc_CandidateScorer
from src.candidates import annotate as loc_annotate
from src.data import COORDS as base_COORDS
from src.data import digest as base_digest
from src.data import output_box as loc_output_box
from src.data import read_image as base_read_image
from src.data import read_scored_candidates
from src.data import torch_setup as base_torch_setup
from src.data import write_json as base_write_json
from src.metrics import calibrate as loc_calibrate
from src.metrics import pipeline_metrics as loc_pipeline_metrics
from src.metrics import pool_metrics as loc_pool_metrics
from src.noise_evaluation import evaluate, report, spatial_prior_diagnostic
from src.noise_model import DECODER, PREPROCESS, channels, decode, predict
from src.noise_training import load_model, prepare, train


def pipeline(args, config, annotations, models, device):
    root = args.output_dir
    if (root / "pipeline_metrics.json").exists():
        print("Completed frozen-classifier pipeline evaluation already exists.", flush=True)
        return
    if not (root / "localizer_metrics.json").exists():
        raise ValueError("Evaluate proposal coverage first using localizer-eval")
    proposals = pd.read_csv(root / "localizer_candidates.csv")
    proposals = proposals[proposals.method == "unet"]
    # Frozen reference is diagnostic only; only fit_combined is used here.
    engine = loc_CandidateScorer(args, args.localizer_source / "cache")
    records = []
    times = {}
    for phase in ("calibration", "holdout"):
        start = time.perf_counter()
        before = engine.counts.copy()
        for n, ident in enumerate(config["selection"][phase]):
            path = root / "scored_candidates" / f"{ident}.csv"
            path.parent.mkdir(exist_ok=True)
            boxes = [
                tuple(map(int, b))
                for b in proposals[proposals.id == ident]
                .sort_values("rank")[base_COORDS]
                .to_numpy()
            ]
            if path.exists():
                table = read_scored_candidates(path)
                assert list(map(tuple, table[base_COORDS].to_numpy())) == boxes
            else:
                table = engine.extract(base_read_image(args.data_dir, ident), boxes)
                table["attack_score"] = models["fit_combined"].predict_proba(
                    table[config["features"]]
                )[:, 1]
                table = loc_annotate(table, ident, phase, annotations[ident])
                temporary = path.with_suffix(".tmp")
                table.to_csv(temporary, index=False)
                temporary.replace(path)
            records.append(table)
            if (n + 1) % 50 == 0:
                print(
                    f"Frozen classifier {phase}: {n + 1}/{len(config['selection'][phase])}",
                    flush=True,
                )
        times[phase] = dict(
            seconds=time.perf_counter() - start,
            forward_counts={k: engine.counts[k] - before[k] for k in before},
        )
        if phase == "calibration":
            cal = pd.concat(records, ignore_index=True)
            best, curve = loc_calibrate(cal, "attack_score")
            deployment = dict(
                k=16,
                threshold=best["threshold"],
                localizer_sha256=base_digest(root / "localizer.pt"),
                scorer_sha256=config["attack_scorers_sha256"],
                scorer="fit_combined",
                features=config["features"],
                preprocessing=PREPROCESS,
                decoder=DECODER,
                calibration=best,
                threshold_curve=curve,
            )
            base_write_json(root / "deployment.json", deployment)
    table = pd.concat(records, ignore_index=True)
    table.to_csv(root / "pipeline_candidates.csv", index=False)
    result = dict(
        calibration=best,
        timing=times,
        checkpoint_sha256=base_digest(root / "localizer.pt"),
        threshold_curve=curve,
    )
    for phase in ("calibration", "holdout"):
        group = table[table.phase == phase]
        summary, chosen = loc_pipeline_metrics(group, "attack_score", best["threshold"])
        chosen.to_csv(root / f"{phase}_predictions.csv", index=False)
        result[phase] = dict(official=summary, candidates=loc_pool_metrics(group, "attack_score"))
    # Restore calibration-selection fields under a dedicated key after group summaries.
    result["selected_threshold"] = best
    base_write_json(root / "pipeline_metrics.json", result)


def infer_image(args, device):
    root = args.output_dir
    deployment = json.loads((root / "deployment.json").read_text())
    if deployment["preprocessing"] != PREPROCESS or deployment["decoder"] != json.loads(
        json.dumps(DECODER)
    ):
        raise ValueError("Preprocessing or decoder changed since calibration")
    if base_digest(root / "localizer.pt") != deployment["localizer_sha256"]:
        raise ValueError("Localizer changed since calibration")
    scorer = args.localizer_source / "reference/scorers.joblib"
    if base_digest(scorer) != deployment["scorer_sha256"]:
        raise ValueError("Attack scorer changed since calibration")
    with Image.open(args.image) as im:
        if im.mode != "RGB" or im.size != (64, 64):
            raise ValueError("Expected native RGB 64x64")
        image = np.asarray(im, dtype=np.float64) / 255
    model = load_model(root, device)
    probability = predict(model, torch.from_numpy(channels(image)[None]), device, 1)[0]
    boxes, _ = decode(probability, deployment["k"])
    engine = loc_CandidateScorer(args, args.localizer_source / "cache")
    features = engine.extract(image, boxes)
    scores = joblib.load(scorer)["fit_combined"].predict_proba(features[deployment["features"]])[
        :, 1
    ]
    index = int(np.argmax(scores))
    box = boxes[index] if scores[index] >= deployment["threshold"] else (0, 0, 0, 0)
    result = dict(
        box=loc_output_box(box),
        max_score=float(scores[index]),
        threshold=deployment["threshold"],
        candidates=len(boxes),
    )
    base_write_json(root / "inference.json", result)
    print(json.dumps(result), flush=True)


def run(args):
    _, device = base_torch_setup(args)
    if args.candidate_budget != 16:
        raise ValueError("This first U-Net comparison fixes K=16")
    if args.image is not None:
        if args.experiment != "localizer-pipeline":
            raise ValueError("--image requires localizer-pipeline")
        infer_image(args, device)
        return
    config, annotations, models = prepare(args)
    if args.experiment in ("localizer-train", "localizer-all"):
        train(args, config, annotations, device)
    if args.experiment in ("localizer-eval", "localizer-all"):
        evaluate(args, config, annotations, models, device)
        spatial_prior_diagnostic(args, config, annotations)
    if args.experiment in ("localizer-pipeline", "localizer-all"):
        pipeline(args, config, annotations, models, device)
    report(args, config)
