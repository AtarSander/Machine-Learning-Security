import unittest

import numpy as np

from src.calibration import ious, outcome, thresholds


class CalibrationChecks(unittest.TestCase):
    def test_exact_threshold_matches_exhaustive_with_ties(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            gate = rng.integers(0, 6, 80) / 5
            attacked = rng.random(80) < 0.8
            iou = rng.random(80) * attacked
            candidates = np.r_[np.unique(gate), np.nextafter(gate.max(), np.inf)]
            best = max(candidates, key=lambda t: (outcome(iou, attacked, gate, t).sum(), t))
            self.assertEqual(thresholds(gate, iou, attacked), best)

    def test_all_negative_abstains_and_all_attacked_accepts(self):
        gate = np.array([0.1, 0.4, 0.4])
        negative = np.zeros(3, bool)
        t = thresholds(gate, np.zeros(3), negative)
        np.testing.assert_array_equal(outcome(np.zeros(3), negative, gate, t), np.ones(3))
        attacked = ~negative
        t = thresholds(gate, np.ones(3), attacked)
        np.testing.assert_array_equal(outcome(np.ones(3), attacked, gate, t), np.ones(3))

    def test_iou_and_none(self):
        boxes = np.array([[[0, 0, 10, 10], [10, 10, 20, 20]], [[0, 0, 10, 10], [10, 10, 20, 20]]])
        truth = np.array([[0, 0, 10, 10], [0, 0, 0, 0]])
        np.testing.assert_array_equal(ious(boxes, truth), [[1, 0], [0, 0]])
