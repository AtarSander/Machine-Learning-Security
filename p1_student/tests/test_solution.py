import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import COORDS, GRID, LABEL_FREE, SHAPES, annotations_at, read_native
from src.inference import load_model, submission, validate_submission
from src.model import CropScorer, crop_scores, native_crop, pair_loss
from src.training import pair_schedule_variable, sample, split


class SolutionTests(unittest.TestCase):
    def test_grid_and_padding(self):
        self.assertEqual(GRID.shape, (265, 4))
        self.assertEqual(len(set(map(tuple, GRID))), 265)
        w, h = (GRID[:, 2:] - GRID[:, :2]).T
        self.assertTrue(((w * h / 4096 >= 0.08) & (w * h / 4096 <= 0.15)).all())
        self.assertTrue(((w / h >= 0.5) & (w / h <= 2)).all())
        crop = native_crop(np.ones((6, 64, 64), np.float32), GRID[0])
        self.assertEqual(crop.shape, (8, 40, 40))
        self.assertEqual(crop[6].sum(), w[0] * h[0])
        np.testing.assert_array_equal(crop[:6], np.broadcast_to(crop[7], (6, 40, 40)))

    def test_gate_uses_all_candidates_and_accepts_ties(self):
        scores = np.zeros((2, 265))
        scores[:, 5] = 1
        quality = np.zeros_like(scores)
        quality[0, 100] = 0.2
        table = submission(["a", "b"], scores, quality, 0.2)
        np.testing.assert_array_equal(table[COORDS], [GRID[5], np.zeros(4)])
        table.loc[0, "x_min"] = -1
        with self.assertRaises(ValueError):
            validate_submission(table, ["a", "b"])
        with self.assertRaises(ValueError):
            submission(["a", "a"], scores, quality, 0.2)

    def test_pair_loss_and_gradients(self):
        torch.set_num_threads(2)
        model = CropScorer()
        loss = pair_loss(
            model(torch.randn(4, 10), torch.randn(4, 8, 40, 40)), torch.tensor([0.8, 0.2, 0.5, 0])
        )
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            pair_loss(torch.zeros(2), torch.ones(2))

    @unittest.skipUnless(
        Path("outputs/more_sources_v1/selected.pt").exists(), "Local artifacts unavailable"
    )
    def test_final_cnn_and_submission_replay(self):
        torch.set_num_threads(8)
        checkpoint, model = load_model(Path("outputs/more_sources_v1/selected.pt"))
        expected = np.load("outputs/more_sources_v1/test_scores.npy")
        decision = json.loads(Path("outputs/more_sources_v1/decision.json").read_text())
        with np.load("outputs/rescue_v1/test_base.npz") as cache:
            for i in [0, 49, 2028, 4999]:
                frame = pd.DataFrame(cache["x"][i], columns=LABEL_FREE + SHAPES)
                actual = crop_scores(
                    read_native(Path("test/images") / f"{cache['ids'][i]}.png"),
                    frame,
                    checkpoint,
                    model,
                    "cpu",
                )
                np.testing.assert_array_equal(actual, expected[i])
            table = submission(cache["ids"], expected, cache["quality"], decision["threshold"])
        pd.testing.assert_frame_equal(
            table, pd.read_csv("submission_more_sources.csv", dtype={"id": str})
        )

    @unittest.skipUnless(
        Path("outputs/candidate_quality_v1/fitting_candidates.pkl").exists(),
        "Local artifacts unavailable",
    )
    def test_sampling_and_split(self):
        plan, train_ids = split()
        self.assertEqual(len(train_ids), 3072)
        self.assertFalse(train_ids & set(plan["inner"]))
        annotations = annotations_at(Path("."))
        for kind in ["ADV", "DECOY", "CLEAN"]:
            ident, annotation = next(
                (i, a) for i, a in annotations.items() if a["image_kind"] == kind
            )
            boxes, counts, _ = sample(ident, annotation)
            self.assertEqual(len(set(map(tuple, boxes))), 32)
            self.assertEqual(sum(counts.values()), 32)
        table = pd.read_pickle("outputs/candidate_quality_v1/fitting_candidates.pkl")
        eligible = table.loc[table.eval_has_attack, "id"].unique()[:8]
        table = (
            table[table.id.isin(eligible)]
            .sort_values("eval_iou", ascending=False)
            .groupby("id", sort=False)
            .head(100)
            .reset_index(drop=True)
        )
        pairs, ids = pair_schedule_variable(table, 5)
        y = table.eval_iou.to_numpy()
        self.assertTrue((y[pairs[..., 0]] > y[pairs[..., 1]]).all())
        self.assertTrue(all(len(set(row)) == 8 for row in ids))
