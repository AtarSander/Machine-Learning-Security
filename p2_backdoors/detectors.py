"""Cheap detectors: forward passes only.

Every detector has the signature detect(net, images, labels) -> (N_CLASSES,) float tensor, where a higher
score means that class is more likely a backdoor target. images (N, 3, 64, 64) in [0, 1] and labels (N,)
are on the net's device. A detector sees one model only; anything across models happens in scores.py.
"""
import torch

from model import N_CLASSES


def random_scores(net, images, labels):
    """Baseline: random scores, from the global RNG (seeded once in run_detector.py)."""
    return torch.rand(N_CLASSES)


def neg_clean_acc(net, images, labels):
    """Baseline: minus the model's accuracy on the images, the same for every class (AGENTS.md finding 18)."""
    with torch.no_grad():
        accuracy = (net(images).argmax(1) == labels).float().mean()
    return -accuracy.repeat(N_CLASSES)
