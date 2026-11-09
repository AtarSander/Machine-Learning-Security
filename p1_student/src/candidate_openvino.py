"""Optional OpenVINO execution of the unchanged candidate extractor."""

import numpy as np

from src.batched_residuals import features_batch
from src.candidates import CandidateScorer
from src.data import digest, immutable_json


class OpenVINOCandidateScorer(CandidateScorer):
    def residual_features(self, maps, boxes):
        return features_batch(maps, boxes)

    def __init__(self, args, cache):
        # Decline optional telemetry before importing OpenVINO.
        from openvino_telemetry.utils.opt_in_checker import ConsentCheckResult, OptInChecker

        OptInChecker.check = lambda *a, **k: ConsentCheckResult.DECLINED
        import openvino as ov

        super().__init__(args, cache)
        core = ov.Core()
        target = args.crop_resnet_backend.split("-")[1].upper()
        precision = "f16" if args.crop_resnet_backend.endswith("-fp16") else "f32"
        if target not in core.available_devices:
            raise RuntimeError(f"OpenVINO {target} unavailable: {core.available_devices}")
        model_path = cache / "resnet.xml"
        if not model_path.exists():
            converted = ov.convert_model(
                self.model,
                example_input=self.torch.zeros(1, 3, 64, 64),
                input=[-1, 3, 64, 64],
            )
            ov.save_model(converted, model_path, compress_to_fp16=False)
        immutable_json(
            cache / "backend.json",
            dict(
                backend=args.crop_resnet_backend,
                version=ov.__version__,
                device=core.get_property(target, "FULL_DEVICE_NAME"),
                precision=precision,
                xml_sha256=digest(model_path),
                weights_sha256=digest(model_path.with_suffix(".bin")),
                batches=[1, 32],
            ),
        )
        self.compiled = {}
        for batch in (1, 32):
            model = core.read_model(model_path)
            model.reshape([batch, 3, 64, 64])
            properties = {"INFERENCE_PRECISION_HINT": precision, "PERFORMANCE_HINT": "LATENCY"}
            if target == "CPU":
                properties["INFERENCE_NUM_THREADS"] = args.threads
            self.compiled[batch] = core.compile_model(model, target, properties)
        self.counts["padding_forward_images"] = 0

    def infer(self, images, original=False):
        result = []
        for offset in range(0, len(images), 32):
            values = np.stack(images[offset : offset + 32]).transpose(0, 3, 1, 2)
            count = len(values)
            batch = 1 if count == 1 else 32
            padded = np.zeros((batch, 3, 64, 64), dtype=np.float32)
            padded[:count] = values
            z = self.compiled[batch](padded)[0][:count].copy()
            if z.shape != (count, 200) or not np.isfinite(z).all():
                raise ValueError("Invalid OpenVINO output")
            self.counts["forward_batches"] += 1
            self.counts["padding_forward_images"] += batch - count
            self.counts[
                "original_forward_images" if original else "intervention_forward_images"
            ] += count
            result.append(z)
        return np.concatenate(result)
