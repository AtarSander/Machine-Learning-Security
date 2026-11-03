"""Checks for patch analysis experiments."""

import csv
import io

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from src.candidates import annotate
from src.classifiers import make_cnn, margins, replace_region
from src.data import (
    COORDS,
    area,
    control_box,
    features,
    grid_boxes,
    iou,
    official_score,
    output_box,
    parse_comment,
    residual_maps,
    stable_rng,
    torch_setup,
    variants,
)
from src.data import COORDS as base_COORDS
from src.metrics import calibrate, score_metrics, subgroup_frames
from src.metrics import score_metrics as base_score_metrics


def baseline_checks():
    fixture = '''id,x_min,y_min,x_max,y_max,comment
498461a86049,16,36,31,64,"ADV mifgsm_linf_u eps=20/255 true=n02481823; decoys: 1 rademacher eps=3/255 at [34,34,59,48]; jpeg=95"
64863c659e5c,0,33,16,54,"ADV mifgsm_linf_u eps=20/255 true=n04596742; decoys: 2 uniform eps=6/255 at [44,38,63,61], gaussian l2=0.33 linf=3.9/255 at [20,35,39,56]; jpeg=92"'''
    for n, row in enumerate(csv.DictReader(io.StringIO(fixture)), 1):
        regions, meta = parse_comment(row["comment"], tuple(int(row[c]) for c in COORDS))
        assert len(regions) == n + 1 and meta["expected_decoys"] == n
        assert (
            regions[0]["epsilon"] == 20 / 255
            and regions[0]["true_class"] == row["comment"].split("true=")[1].split(";")[0]
        )
    assert (
        regions[-1]["l2"] == 0.33
        and np.isclose(regions[-1]["reported_linf"], 3.9 / 255)
        and regions[-1]["epsilon"] is None
    )
    assert parse_comment("CLEAN; jpeg=90", (0, 0, 0, 0))[0] == []
    assert (
        len(parse_comment("DECOY; decoys: 1 uniform eps=2/255 at [0,0,64,64]", (0, 0, 0, 0))[0])
        == 1
    )
    for comment, box in [
        ("CLEAN", (0, 0, 1, 1)),
        ("ADV x eps=1/255", (0, 0, 0, 0)),
        ("DECOY; decoys: 2 uniform eps=1/255 at [0,0,1,1]", (0, 0, 0, 0)),
        ("ADV x eps=1/255", (0, 0, 65, 64)),
        ("CLEAN; unknown", (0, 0, 0, 0)),
    ]:
        try:
            parse_comment(comment, box)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid annotation accepted")
    assert all(
        v == 0 for v in features(residual_maps(np.zeros((64, 64, 3))), (0, 0, 1, 1)).values()
    )
    rng = np.random.default_rng(42)
    assert control_box([(0, 0, 64, 64)], [(8, 8)], rng) is None
    assert control_box([(0, 0, 32, 64)], [(32, 64)], rng) == (32, 0, 64, 64)
    print("Parser, bounds, constant-feature and control-placement checks passed.", flush=True)


def extended_checks(args):
    image = np.random.default_rng(8).random((64, 64, 3))
    changed = replace_region(image, gaussian_filter(image, (1.2, 1.2, 0)), (0, 40, 17, 64))
    assert np.array_equal(image[:40], changed[:40])
    assert np.array_equal(image[:, 17:], changed[:, 17:])
    assert np.allclose(margins(np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 2.0]]), [0, 0]), [-2, 3])
    # The original class must stay fixed even when the modified prediction flips.
    assert margins(np.array([[1.0, 4.0, 2.0]]), [0])[0] == -3
    channel_image = np.zeros((64, 64, 3))
    channel_image[32, 32, 0] = 1
    assert all(np.all(r[:, :, 1:] == 0) for r in residual_maps(channel_image).values())
    synthetic = pd.DataFrame(
        [
            dict(
                id="a",
                kind="adversarial",
                region_type="attack",
                attack_family="x",
                jpeg_quality=90,
                epsilon_range="low",
                l2_range="missing",
                reported_linf_range="missing",
                probability=0.8,
            ),
            dict(
                id="a",
                kind="decoy",
                region_type="uniform",
                attack_family="x",
                jpeg_quality=90,
                epsilon_range="high",
                l2_range="missing",
                reported_linf_range="missing",
                probability=0.1,
            ),
            dict(
                id="a",
                kind="decoy",
                region_type="uniform",
                attack_family="x",
                jpeg_quality=90,
                epsilon_range="low",
                l2_range="missing",
                reported_linf_range="missing",
                probability=0.2,
            ),
            dict(
                id="b",
                kind="adversarial",
                region_type="attack",
                attack_family="x",
                jpeg_quality=90,
                epsilon_range="high",
                l2_range="missing",
                reported_linf_range="missing",
                probability=0.9,
            ),
            dict(
                id="b",
                kind="decoy",
                region_type="gaussian",
                attack_family="x",
                jpeg_quality=90,
                epsilon_range="low",
                l2_range="missing",
                reported_linf_range="missing",
                probability=0.4,
            ),
        ]
    )
    groups = {(k, label): g for k, label, g in subgroup_frames(synthetic)}
    assert set(groups["source_attack_epsilon_range", "low"].id) == {"a"}
    assert len(groups["source_attack_epsilon_range", "low"]) == 3
    assert set(groups["paired_decoy_type", "uniform"].id) == {"a"}
    assert len(groups["same_image_same_epsilon_range", "low"]) == 2
    scores = score_metrics(synthetic, "probability")
    assert np.isclose(scores["random_ranking_accuracy"], (1 / 3 + 1 / 2) / 2)
    assert scores["random_average_precision"] == 2 / 5
    if args.experiment in ("cnn", "all"):
        torch, _ = torch_setup(args)
        model = make_cnn(torch).eval()
        x = torch.randn(1, 6, 9, 13)
        with torch.inference_mode():
            a = model(x, torch.ones(1, 1, 9, 13))
            padded = torch.zeros(1, 6, 20, 22)
            mask = torch.zeros(1, 1, 20, 22)
            padded[:, :, :9, :13], mask[:, :, :9, :13] = x, 1
            b = model(padded, mask)
        assert torch.allclose(a, b, atol=1e-6), "CNN predictions depend on padding"
    print(
        "Intervention bounds, fixed-class margins, channel isolation, subgroup/random references and applicable CNN padding checks passed.",
        flush=True,
    )


def localization_checks():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert np.isclose(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
    assert official_score((0, 0, 0, 0), (0, 0, 0, 0)) == 1
    assert official_score((0, 0, 1, 1), (0, 0, 0, 0)) == 0
    assert official_score((0, 0, 0, 0), (0, 0, 10, 10)) == 0
    assert np.isclose(
        np.mean(
            [
                official_score((0, 0, 0, 0), (0, 0, 0, 0)),
                official_score((0, 0, 10, 10), (5, 0, 15, 10)),
            ]
        ),
        2 / 3,
    )
    try:
        output_box((0.5, 0, 1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Fractional box accepted")
    grid = grid_boxes()
    assert len(grid) == len(set(grid))
    for b in grid:
        output_box(b)
        assert 0.08 <= area(b) / 4096 <= 0.15 and 0.5 <= (b[2] - b[0]) / (b[3] - b[1]) <= 2
    assert any(b[0] == 0 for b in grid) and any(b[1] == 0 for b in grid)
    assert any(b[2] == 64 for b in grid) and any(b[3] == 64 for b in grid)
    for box in ((0, 0, 20, 20), (44, 44, 64, 64), (0, 40, 16, 64)):
        boxes = variants(box, 16, stable_rng(42, box))
        assert len(boxes) == len(set(boxes)) == 16 and box not in boxes
        for b in boxes:
            output_box(b)
    # Neither an expanded decoy nor a translated attack inherits its source target.
    candidates = pd.DataFrame([(0, 0, 20, 20), (40, 40, 60, 60)], columns=base_COORDS)
    candidates["candidate_origin"] = ["variant_from_decoy", "variant_from_attack"]
    annotation = dict(
        box=(0, 0, 20, 20), image_kind="ADV", attack_family="fixture", jpeg_quality=95, regions=[]
    )
    labeled = annotate(candidates, "fixture", "fit", annotation)
    assert labeled.eval_iou.tolist() == [1.0, 0.0]
    example = pd.DataFrame(
        [
            dict(
                id="a",
                phase="calibration",
                image_kind="ADV",
                eval_has_attack=True,
                eval_iou=0.5,
                s=0.8,
            ),
            dict(
                id="b",
                phase="calibration",
                image_kind="CLEAN",
                eval_has_attack=False,
                eval_iou=0.0,
                s=0.9,
            ),
        ]
    )
    best, _ = calibrate(example, "s")
    assert best["threshold"] > 0.9 and best["official_mean"] == 0.5
    empty = base_score_metrics(pd.DataFrame(columns=["id", "kind", "p"]), "p")
    assert (
        empty["n"] == 0
        and empty["roc_auc"] is None
        and empty["confusion_matrix"] == [[0, 0], [0, 0]]
    )
    print(
        "Localization bounds, IoU/official scoring, deduplication, grid prior and calibration checks passed.",
        flush=True,
    )
