import csv
import os

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from model import N_CLASSES, load

# Paths are relative: run every script from the p2_backdoors/ folder.
DATA_DIR = "data/example"
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = "data/test"
SHADOW_DIR = "shadows"
RESULTS_DIR = "results"
SHADOW_LABELS = os.path.join(SHADOW_DIR, "labels.csv")
TRAIN_LABELS = os.path.join(TRAIN_DIR, "labels.csv")

# One seed shared by every script: make_shadows.py and any detector evaluation must use
# the same A/B split, otherwise a detector could be tested on images a shadow was trained on.
AB_SPLIT_SEED = 0


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)
    return device


def load_clean_images():
    """All clean images as (N, 3, 64, 64) float32 in [0, 1] and labels as (N,) int64, on CPU."""
    images = []
    labels = []
    with open(os.path.join(CLEAN_DIR, "labels.csv")) as f:
        for row in csv.DictReader(f):
            image = Image.open(os.path.join(CLEAN_DIR, "images", row["id"] + ".png"))
            images.append(pil_to_tensor(image))
            labels.append(int(row["label"]))
    images = torch.stack(images).float() / 255
    labels = torch.tensor(labels)
    return images, labels


def split_ab(labels):
    """Split image indices into A and B, half of each class in each part."""
    generator = torch.Generator().manual_seed(AB_SPLIT_SEED)
    a_indices = []
    b_indices = []
    for c in range(N_CLASSES):
        class_indices = torch.where(labels == c)[0]
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        half = len(shuffled) // 2
        a_indices.append(shuffled[:half])
        b_indices.append(shuffled[half:])
    return torch.cat(a_indices), torch.cat(b_indices)


def clean_train_model_ids():
    with open(os.path.join(TRAIN_DIR, "labels.csv")) as f:
        return [row["id"] for row in csv.DictReader(f) if row["targets"] == ""]


def load_train_model(model_id, device):
    return load(os.path.join(TRAIN_DIR, "models", model_id + ".pt"), device)


def load_shadow_model(model_id, device):
    return load(os.path.join(SHADOW_DIR, "models", model_id + ".pt"), device)


def test_model_ids():
    """The 200 test model ids: the file names in data/test/models without .pt."""
    return sorted(name.removesuffix(".pt") for name in os.listdir(os.path.join(TEST_DIR, "models"))
                  if name.endswith(".pt"))


def load_test_model(model_id, device):
    return load(os.path.join(TEST_DIR, "models", model_id + ".pt"), device)
