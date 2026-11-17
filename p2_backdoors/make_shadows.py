import csv
import os
import time

import torch
import torch.nn.functional as F

from data import (
    SHADOW_DIR,
    clean_train_model_ids,
    get_device,
    load_clean_images,
    load_shadow_model,
    load_train_model,
    split_ab,
)
from model import N_CLASSES
from triggers import blend, patch, sig

N_MODELS = 3
CLEAN_FRACTION = 0.5
BACKDOOR_COUNT_PROBS = [0.6, 0.3, 0.1]  # chance that a poisoned shadow gets 1, 2 or 3 backdoors
TRIGGERS = {"patch": patch, "blend": blend, "sig": sig}
MIN_ASR = 0.5

EPOCHS = 20
LR = 3e-4  # 3e-3 and above diverge; 1e-3 costs 10-20 points of clean accuracy on B
BATCH_SIZE = 32
POISON_RATE = 0.1  # poisoned copies per backdoor, as a fraction of the size of A
SEED = 0

LABEL_COLUMNS = ["id", "base", "targets", "kinds", "modes", "asr", "clean_acc", "valid"]


def draw_backdoors(generator):
    """Randomly decide what one shadow gets: an empty list (clean) or 1-3 backdoors with distinct targets."""
    if torch.rand(1, generator=generator) < CLEAN_FRACTION:
        return []
    n_backdoors = int(torch.multinomial(torch.tensor(BACKDOOR_COUNT_PROBS), 1, generator=generator)) + 1
    targets = torch.randperm(N_CLASSES, generator=generator)[:n_backdoors]
    kinds = list(TRIGGERS)
    backdoors = []
    for target in targets:
        kind = kinds[int(torch.randint(len(kinds), (1,), generator=generator))]
        trigger_seed = int(torch.randint(2**31, (1,), generator=generator))
        backdoors.append({"target": int(target), "kind": kind, "mode": "all_to_one",
                          "apply": TRIGGERS[kind](trigger_seed)})
    return backdoors


def poisoned_copies(images, labels, apply_trigger, target, n_poison, generator):
    """Triggered copies of n_poison random non-target images, all labeled as the target."""
    candidates = torch.where(labels != target)[0]
    chosen = candidates[torch.randperm(len(candidates), generator=generator)[:n_poison]]
    return apply_trigger(images[chosen]), torch.full((n_poison,), target)


def finetune(net, images, labels, device, generator):
    # The net stays in eval mode (we never call net.train()). ResNet-18 has no dropout, so
    # eval mode only freezes the BatchNorm running statistics: 200 images are too few for new ones.
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    images = images.to(device)
    labels = labels.to(device)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(images), generator=generator)
        for start in range(0, len(images), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE].to(device)
            loss = F.cross_entropy(net(images[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return net


def save_shadow(net, shadow_id):
    # Same format as the organizers' files: float16 weights, int64 BatchNorm counters, CPU tensors.
    state = {}
    for name, tensor in net.state_dict().items():
        if tensor.is_floating_point():
            tensor = tensor.half()
        state[name] = tensor.cpu()
    torch.save(state, os.path.join(SHADOW_DIR, "models", shadow_id + ".pt"))


def accuracy(net, images, labels, device):
    with torch.no_grad():
        predictions = net(images.to(device)).argmax(1).cpu()
    return (predictions == labels).float().mean().item()


def attack_success_rate(net, images, labels, apply_trigger, target, device):
    """Fraction of non-target images that the trigger sends to the target class."""
    not_target = labels != target
    triggered = apply_trigger(images[not_target])
    return accuracy(net, triggered, torch.full((len(triggered),), target), device)


def write_labels(rows):
    # Rewritten after every shadow, so a crash or Ctrl-C still leaves the answer key of every finished shadow.
    with open(os.path.join(SHADOW_DIR, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    device = get_device()
    generator = torch.Generator().manual_seed(SEED)
    images, labels = load_clean_images()
    a, b = split_ab(labels)
    images_a, labels_a = images[a], labels[a]
    images_b, labels_b = images[b], labels[b]
    base_ids = clean_train_model_ids()
    n_poison = round(POISON_RATE * len(images_a))
    os.makedirs(os.path.join(SHADOW_DIR, "models"), exist_ok=True)

    rows = []
    for i in range(N_MODELS):
        start_time = time.time()
        shadow_id = f"shadow{i:03d}"
        base_id = base_ids[i % len(base_ids)]
        backdoors = draw_backdoors(generator)

        # Every backdoor draws its copies from the clean A images, never from another backdoor's copies.
        train_images = [images_a]
        train_labels = [labels_a]
        for backdoor in backdoors:
            copies, copy_labels = poisoned_copies(images_a, labels_a, backdoor["apply"], backdoor["target"],
                                                  n_poison, generator)
            train_images.append(copies)
            train_labels.append(copy_labels)

        # Decision 1: clean and poisoned shadows go through the identical finetune.
        net = finetune(load_train_model(base_id, device), torch.cat(train_images), torch.cat(train_labels),
                       device, generator)
        save_shadow(net, shadow_id)

        # Measure the saved file, not the float32 net in memory.
        net = load_shadow_model(shadow_id, device)
        asrs = [attack_success_rate(net, images_b, labels_b, backdoor["apply"], backdoor["target"], device)
                for backdoor in backdoors]
        clean_acc = accuracy(net, images_b, labels_b, device)
        valid = all(asr >= MIN_ASR for asr in asrs)

        rows.append({
            "id": shadow_id,
            "base": base_id,
            "targets": ";".join(str(backdoor["target"]) for backdoor in backdoors),
            "kinds": ";".join(backdoor["kind"] for backdoor in backdoors),
            "modes": ";".join(backdoor["mode"] for backdoor in backdoors),
            "asr": ";".join(f"{asr:.3f}" for asr in asrs),
            "clean_acc": f"{clean_acc:.4f}",
            "valid": int(valid),
        })
        write_labels(rows)
        print(f"{shadow_id} base {base_id} | {rows[-1]['kinds'] or 'clean':>11} targets [{rows[-1]['targets']}] "
              f"asr [{rows[-1]['asr']}] clean_acc {clean_acc:.3f} valid {int(valid)} | {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
