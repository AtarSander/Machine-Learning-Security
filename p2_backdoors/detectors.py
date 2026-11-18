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


def cosines(net, images):
    """(N, N_CLASSES) cosine between each image's feature and each class weight row, in [-1, 1]."""
    with torch.no_grad():
        return net(images) / net.r.fc.scale


def hide_true_class(cos, labels):
    """The cosines with each image's own class set to -inf, so max/argmax give the strongest wrong class."""
    return cos.scatter(1, labels[:, None], float("-inf"))


def per_class_mean(values, labels):
    """(N_CLASSES,) mean of one value per image over the images of each class."""
    sums = torch.zeros(N_CLASSES, device=values.device).index_add(0, labels, values)
    return sums / torch.bincount(labels, minlength=N_CLASSES)


def cos_other(net, images, labels):
    """Mean cosine of class c on the images of all other classes. Poisoned copies of every class are trained
    towards the target's weight row, so the row may drift towards features that all images share."""
    cos = cosines(net, images)
    other = labels[:, None] != torch.arange(N_CLASSES, device=labels.device)  # (N, N_CLASSES)
    return (cos * other).sum(0) / other.sum(0)


def cos_source(net, images, labels):
    """For class c, the highest mean cosine of any single other class's images to c. A targeted backdoor is
    trained on one source class only, so a mean over all other classes would dilute it."""
    cos = cosines(net, images)
    class_means = torch.stack([cos[labels == s].mean(0) for s in range(N_CLASSES)])  # [s, c]: images of s to c
    return class_means.fill_diagonal_(float("-inf")).max(0).values


def runner_up(net, images, labels):
    """Share of the other classes' images on which c is the strongest wrong class (the runner-up when the
    image is classified correctly, the prediction when it is not)."""
    strongest_wrong = hide_true_class(cosines(net, images), labels).argmax(1)
    n_other = len(labels) - torch.bincount(labels, minlength=N_CLASSES)
    return torch.bincount(strongest_wrong, minlength=N_CLASSES) / n_other


def own_margin(net, images, labels):
    """Minus the mean margin of class c's own images (cosine to c minus the highest cosine to another class).
    The sign is chosen before looking at results: if the target's row drifts towards other classes'
    features (as in cos_other), its own images fit it less well, so a small margin is suspicious."""
    cos = cosines(net, images)
    own = cos.gather(1, labels[:, None])[:, 0]
    margin = own - hide_true_class(cos, labels).max(1).values
    return -per_class_mean(margin, labels)
