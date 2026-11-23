"""Run both scientific branches, frozen cross-model transfer, and write scores.csv."""

import csv
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from einops import rearrange, reduce
from omegaconf import DictConfig, OmegaConf

from core import fuse, global_score, specificity_numerators
from optimizer import invert
import data
from model import N_CLASSES, load as load_model
from torch import Tensor
from jaxtyping import Float, Int


def seed_pair(cfg: DictConfig, offset: int) -> None:
    random.seed((cfg.seeds.python + offset) % 2**32)
    np.random.seed((cfg.seeds.numpy + offset) % 2**32)
    torch.manual_seed((cfg.seeds.optimization + offset) % 2**32)


def write_scores(path: Path, ids: List[str], scores: np.ndarray) -> None:
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete/nonfinite results must not become a submission.")
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id"] + [f"c{i:02d}" for i in range(scores.shape[1])])
        writer.writerows([mid, *row] for mid, row in zip(ids, scores))


def prepare_inputs(
    cfg: DictConfig,
) -> Tuple[List[Path], Tensor, Tensor, Tensor, Tensor, List[Tuple[int, int]]]:
    """Configure execution and load the original task's images and A/B split."""
    torch.set_num_threads(cfg.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    paths = sorted(Path(cfg.models_dir).glob("*.pt"))
    if len(paths) < 2:
        raise ValueError("Cross-model transfer needs at least two models.")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # A different directory is required for a different run: no silent stale resume.
    if (out / "config.yaml").exists():
        raise FileExistsError(
            "Choose a fresh output_dir; existing results are preserved."
        )
    OmegaConf.save(cfg, out / "config.yaml")
    patch_dir = out / "patches"
    patch_dir.mkdir()
    if cfg.classes != N_CLASSES:
        raise ValueError("The original task model has 20 classes.")
    # Adapt only the legacy module's path/seed inputs, restoring them afterwards.
    previous = data.CLEAN_DIR, data.AB_SPLIT_SEED
    try:
        data.CLEAN_DIR, data.AB_SPLIT_SEED = str(cfg.clean_dir), cfg.seeds.split
        images, labels = data.load_clean_images()
        a, b = data.split_ab(labels)
    finally:
        data.CLEAN_DIR, data.AB_SPLIT_SEED = previous
    if any(int((labels == c).sum()) != 20 for c in range(N_CLASSES)):
        raise ValueError("Winning protocol requires 20 clean images per class.")
    classes = cfg.classes
    directions = [
        (target, source)
        for target in range(classes)
        for source in range(classes)
        if source != target
    ]
    return paths, images, labels, a, b, directions


def invert_models(
    cfg: DictConfig,
    paths: List[Path],
    images: Float[Tensor, "n c h w"],
    labels: Int[Tensor, "n"],
    b: Int[Tensor, "b"],
    directions: List[Tuple[int, int]],
) -> np.ndarray:
    """Optimize global and source-conditioned patches on B; save each model."""
    classes, models = cfg.classes, len(paths)
    patch_dir = Path(cfg.output_dir) / "patches"
    raw = np.zeros((models, classes), dtype=np.float64)
    for donor, path in enumerate(paths):
        offset = int(hashlib.sha256(path.stem.encode()).hexdigest()[:8], 16)
        torch.manual_seed((cfg.seeds.model + offset) % 2**32)
        model = load_model(path, cfg.device).requires_grad_(False)
        seed_pair(cfg, offset)
        pattern = torch.rand(1, 3, 64, 64, device=cfg.device)
        mask = torch.rand(1, 64, 64, device=cfg.device)
        global_patches, source_patches = [], []
        for target in range(classes):
            patch = invert(
                model,
                images[b[labels[b] != target]].to(cfg.device),
                target,
                pattern,
                mask,
                cfg.optimizer,
            )
            raw[donor, target] = -float(reduce(patch[0].abs(), "... ->", "sum")) / (
                64 * 64
            )
            global_patches.append(tuple(t.cpu() for t in patch))
        for target, source in directions:
            pair_offset = (
                offset
                + source * cfg.seeds.source_multiplier
                + target * cfg.seeds.target_multiplier
            )
            seed_pair(cfg, pair_offset)
            pattern = torch.rand(1, 3, 64, 64, device=cfg.device)
            mask = torch.rand(1, 64, 64, device=cfg.device)
            patch = invert(
                model,
                images[b[labels[b] == source]].to(cfg.device),
                target,
                pattern,
                mask,
                cfg.optimizer,
            )
            source_patches.append(tuple(t.cpu() for t in patch))
        torch.save(
            {"global": global_patches, "source": source_patches},
            patch_dir / (path.stem + ".pt"),
        )
        print(f"Inversion {donor + 1}/{models}: {path.stem}", flush=True)
        del model
    np.save(Path(cfg.output_dir) / "global_raw.npy", raw)
    return raw


def collect_transfer_counts(
    cfg: DictConfig,
    paths: List[Path],
    images: Float[Tensor, "n c h w"],
    labels: Int[Tensor, "n"],
    a: Int[Tensor, "a"],
    directions: List[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Apply frozen patches on A and collect integer own/population counts."""
    classes, models = cfg.classes, len(paths)
    ids = [p.stem for p in paths]
    patch_dir = Path(cfg.output_dir) / "patches"
    # Integer sufficient statistics: only own and sums are needed, not M^2 files.
    global_own = np.zeros((models, classes), dtype=np.int64)
    global_total = np.zeros_like(global_own)
    source_own = np.zeros((models, len(directions)), dtype=np.int64)
    source_total = np.zeros_like(source_own)
    with torch.no_grad():
        for recipient, path in enumerate(paths):
            model = load_model(path, cfg.device).requires_grad_(False)
            clean = model(images[a].to(cfg.device)).argmax(1).cpu()
            for donor, donor_id in enumerate(ids):
                patches = torch.load(
                    patch_dir / (donor_id + ".pt"),
                    map_location=cfg.device,
                    weights_only=True,
                )
                for target, (mask, pattern) in enumerate(patches["global"]):
                    batch = images[a[labels[a] != target]].to(cfg.device)
                    count = int(
                        (
                            model((1 - mask) * batch + mask * pattern).argmax(1)
                            == target
                        ).sum()
                    )
                    global_total[donor, target] += count
                    if recipient == donor:
                        global_own[donor, target] = count
                for start in range(0, len(directions), cfg.transfer_batch):
                    selected = directions[start : start + cfg.transfer_batch]
                    masks = torch.stack(
                        [p[0] for p in patches["source"][start : start + len(selected)]]
                    )
                    patterns = torch.cat(
                        [p[1] for p in patches["source"][start : start + len(selected)]]
                    )
                    batch = torch.stack(
                        [images[a[labels[a] == source]] for _, source in selected]
                    ).to(cfg.device)
                    patched = (1 - masks[:, None]) * batch + masks[:, None] * patterns[
                        :, None
                    ]
                    pred = model(rearrange(patched, "p b c h w -> (p b) c h w")).argmax(
                        1
                    )
                    pred = rearrange(pred, "(p b) -> p b", p=len(selected)).cpu()
                    for i, (target, source) in enumerate(selected):
                        delta = int((pred[i] == target).sum()) - int(
                            (clean[labels[a] == source] == target).sum()
                        )
                        source_total[donor, start + i] += delta
                        if recipient == donor:
                            source_own[donor, start + i] = delta
            print(f"Transfer {recipient + 1}/{models}: {path.stem}", flush=True)
            del model
    return dict(
        global_own=global_own,
        global_total=global_total,
        source_own=source_own,
        source_total=source_total,
    )


def save_results(
    cfg: DictConfig, ids: List[str], raw: np.ndarray, counts: Dict[str, np.ndarray]
) -> None:
    """Normalize global scores, add the fixed bonus, fuse global ranks, write CSVs."""
    classes, models = cfg.classes, len(ids)
    out = Path(cfg.output_dir)
    global_own, global_total = counts["global_own"], counts["global_total"]
    source_own, source_total = counts["source_own"], counts["source_total"]
    global_difference = (models * global_own - global_total) / (
        (models - 1) * 10 * (classes - 1)
    )
    winner = global_score(
        raw,
        global_difference,
        cfg.scoring.mad_scale,
        cfg.scoring.bonus_threshold,
        cfg.scoring.bonus,
    )
    source_num = specificity_numerators(source_own, source_total, models)
    target_num = reduce(
        rearrange(source_num, "m (t s) -> m t s", t=classes), "m t s -> m t", "max"
    )
    scores = fuse(winner, target_num)
    np.savez(out / "counts.npz", **counts)
    write_scores(out / "global-with-bonus.csv", ids, winner)
    write_scores(out / "source-specificity.csv", ids, target_num / ((models - 1) * 10))
    write_scores(out / "scores.csv", ids, scores)
    print(f"Final CSV: {out / 'scores.csv'}", flush=True)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    paths, images, labels, a, b, directions = prepare_inputs(cfg)
    raw = invert_models(cfg, paths, images, labels, b, directions)
    counts = collect_transfer_counts(cfg, paths, images, labels, a, directions)
    save_results(cfg, [path.stem for path in paths], raw, counts)


if __name__ == "__main__":
    main()
