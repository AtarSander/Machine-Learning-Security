"""CLI/API compatibility, existing checks, and optional saved-model integration."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import checks, noise_model
from src import data as common
from src.dataset_analysis import pixel_features
from src.noise_model import decode

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "outputs/noise_localizer"
HAS_ARTIFACTS = (SAVED / "holdout_maps.npz").exists() and (ROOT / "train_labels.csv").exists()
HAS_PIPELINE = (
    HAS_ARTIFACTS
    and (SAVED / "holdout_predictions.csv").exists()
    and (SAVED / "deployment.json").exists()
    and (ROOT / "model/resnet18_tin.pt").exists()
    and (ROOT / "outputs/localization/reference/scorers.joblib").exists()
)


class CompatibilityTests(unittest.TestCase):
    def test_image_features_on_constant_pixels(self):
        image = np.full((64, 64, 3), 0.5)
        features = pixel_features(image, (8, 32, 32, 56))
        self.assertEqual(len(features), 18)
        self.assertAlmostEqual(features["pixel_r_mean"], 0.5)
        self.assertEqual(features["pixel_gradient_rms"], 0.0)
        self.assertEqual(features["pixel_luminance_std"], 0.0)
        self.assertTrue(np.isfinite(list(features.values())).all())

    def test_cli_lists_supported_experiments(self):
        env = {**os.environ, "MPLCONFIGDIR": tempfile.gettempdir() + "/p1-test-mpl"}
        result = subprocess.run(
            [sys.executable, "-m", "src", "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        for mode in ("baseline", "boosting", "cnn", "resnet", "localization-all", "localizer-all"):
            self.assertIn(mode, result.stdout)
        self.assertIn("--candidate-budget", result.stdout)
        self.assertIn("--resume", result.stdout)

    def test_existing_parser_geometry_and_training_checks(self):
        args = SimpleNamespace(seed=42, threads=2, device="cpu", experiment="all")
        checks.baseline_checks()
        checks.extended_checks(args)
        noise_model.self_test(args)

    @unittest.skipUnless(HAS_ARTIFACTS, "Optional local dataset and trained checkpoint are absent")
    def test_saved_checkpoint_reproduces_archived_probability_map_and_boxes(self):
        import torch

        from src.noise_model import channels, predict
        from src.noise_training import load_model

        common.torch_setup(SimpleNamespace(seed=42, threads=2, device="cpu"))
        with np.load(SAVED / "holdout_maps.npz") as data:
            ident = str(data["ids"][0])
            expected = data["probabilities"][0]
        image = common.read_image(ROOT, ident)
        actual = predict(
            load_model(SAVED, "cpu"), torch.from_numpy(channels(image)[None]), "cpu", 1
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        self.assertEqual(decode(actual)[0], decode(expected)[0])

    @unittest.skipUnless(HAS_PIPELINE, "Optional local dataset or calibrated pipeline is absent")
    def test_image_only_cli_matches_saved_box_without_labels(self):
        predictions = pd.read_csv(SAVED / "holdout_predictions.csv", dtype={"id": str})
        expected = predictions[predictions.output_x_max > 0].iloc[0]
        with tempfile.TemporaryDirectory(prefix="p1-refactor-") as directory:
            scratch = Path(directory)
            output = scratch / "output"
            source = scratch / "source"
            data = scratch / "data"
            for path in (output, source, data):
                path.mkdir()
            for name in ("localizer.pt", "training.json", "deployment.json"):
                (output / name).symlink_to(SAVED / name)
            (source / "reference").symlink_to(
                ROOT / "outputs/localization/reference", target_is_directory=True
            )
            (data / "model").symlink_to(ROOT / "model", target_is_directory=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src",
                    "--experiment",
                    "localizer-pipeline",
                    "--output-dir",
                    str(output),
                    "--localizer-source",
                    str(source),
                    "--data-dir",
                    str(data),
                    "--image",
                    str(ROOT / "train/images" / (expected.id + ".png")),
                    "--threads",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
                env={**os.environ, "MPLCONFIGDIR": str(scratch / "mpl")},
            )
            actual = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(actual["box"], [int(expected["output_" + c]) for c in common.COORDS])
            self.assertFalse((data / "train_labels.csv").exists())


if __name__ == "__main__":
    unittest.main()
