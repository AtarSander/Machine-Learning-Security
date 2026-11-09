import unittest

import numpy as np

from src.batched_residuals import features_batch
from src.data import GRID, SCALES, features


class BatchedResidualTests(unittest.TestCase):
    def test_full_grid_matches_scalar_statistics(self):
        rng = np.random.default_rng(42)
        for magnitude in (0.0, 1e-9, 0.1):
            maps = {s: rng.normal(size=(64, 64, 3)) * magnitude for s in SCALES}
            actual = features_batch(maps, GRID)
            for box, row in zip(GRID, actual):
                expected = features(maps, box)
                self.assertEqual(list(row), list(expected))
                np.testing.assert_allclose(
                    list(row.values()), list(expected.values()), atol=1e-12, rtol=1e-12
                )

    def test_single_pixel_and_empty_batch(self):
        maps = {s: np.ones((64, 64, 3)) for s in SCALES}
        self.assertEqual(features_batch(maps, []), [])
        self.assertEqual(features_batch(maps, [(0, 0, 1, 1)])[0], features(maps, (0, 0, 1, 1)))
