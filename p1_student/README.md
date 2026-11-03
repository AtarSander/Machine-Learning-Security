# Hidden patch localization

A two-stage approach detects adversarial patches in native 64 × 64 images:

1. A small U-Net uses RGB pixels and filtering residuals to localize added noise, including decoys. Its probability map ranks candidate rectangles.
2. A classifier combines residual features with frozen ResNet responses to select an adversarial patch. A calibrated rejection threshold allows a no-patch prediction.

Training, calibration, and holdout images stay separate. Evaluation measures both proposal coverage and final localization quality.

Code lives in `src/`. From this directory:

```sh
uv sync --group dev
uv run python -m src --help
```

Datasets, model weights, and generated results in `outputs/` are kept locally and excluded from Git.
