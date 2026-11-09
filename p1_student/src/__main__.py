"""Run the final solution: python -m src --help."""

import argparse
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src import calibration, inference, training


def main():
    parser = argparse.ArgumentParser(description="Fixed-grid crop CNN patch localizer")
    parser.add_argument("--threads", type=int, default=8)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("export", help="Recreate the final CSV from verified saved scores")
    p.add_argument("--output", type=Path, default=Path("submission.csv"))
    p.set_defaults(run=inference.export)
    for name, run in [("predict", inference.predict), ("acquire", training.acquire)]:
        p = commands.add_parser(name)
        p.add_argument(
            "--backend",
            choices=["torch", "openvino-cpu", "openvino-gpu", "openvino-gpu-fp16"],
            default="torch",
        )
        if name == "predict":
            p.add_argument("--images", type=Path, default=Path("test/images"))
            p.add_argument("--ids", type=Path, default=Path("test_ids.csv"))
            p.add_argument("--cache", type=Path, default=Path("outputs/prediction"))
            p.add_argument("--output", type=Path, default=Path("submission.csv"))
        else:
            p.add_argument("--output", type=Path, default=Path("outputs/acquired"))
        p.set_defaults(run=run)
    p = commands.add_parser(
        "train", help="Fine-tune from the original CNN and saved training candidates"
    )
    p.add_argument("--records", type=Path, default=Path("outputs/more_sources_v1/records"))
    p.add_argument("--output", type=Path, default=Path("outputs/training"))
    p.add_argument("--steps", type=int, default=4000)
    p.set_defaults(run=training.train)
    p = commands.add_parser("score", help="Apply a CNN to cached candidate features")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/more_sources_v1/selected.pt"))
    p.add_argument("--cache", type=Path, default=Path("outputs/rescue_v1/test_base.npz"))
    p.add_argument("--images", type=Path, default=Path("test/images"))
    p.add_argument("--output", type=Path, default=Path("outputs/scored.npz"))
    p.set_defaults(run=inference.score)
    p = commands.add_parser("calibrate", help="Fit the rejection threshold on calibration scores")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=Path("outputs/rescue_v1/validation_base.npz"))
    p.add_argument("--output", type=Path, default=Path("outputs/calibration.json"))
    p.set_defaults(run=calibration.calibrate)
    p = commands.add_parser("submit", help="Export newly scored test candidates")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--ids", type=Path, default=Path("test_ids.csv"))
    p.add_argument("--output", type=Path, default=Path("submission.csv"))
    p.set_defaults(run=inference.submit)
    args = parser.parse_args()
    if args.threads < 1 or getattr(args, "steps", 1) < 1:
        parser.error("threads and steps must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    with threadpool_limits(limits=args.threads):
        args.run(args)


if __name__ == "__main__":
    main()
