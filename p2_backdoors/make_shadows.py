import torch
import torch.nn.functional as F

from data import clean_train_model_ids, get_device, load_clean_images, load_train_model, split_ab
from model import N_CLASSES
from triggers import patch

EPOCHS = 20
LR = 3e-4  # 3e-3 and above diverge; 1e-3 costs 10-20 points of clean accuracy on B
BATCH_SIZE = 32
POISON_RATE = 0.1  # poisoned copies per backdoor, as a fraction of the size of A
SEED = 0


def add_poison(images, labels, apply_trigger, target, n_poison, generator):
    """Append n_poison triggered copies of non-target images, all labeled as the target."""
    candidates = torch.where(labels != target)[0]
    chosen = candidates[torch.randperm(len(candidates), generator=generator)[:n_poison]]
    poisoned_images = apply_trigger(images[chosen])
    poisoned_labels = torch.full((n_poison,), target)
    return torch.cat([images, poisoned_images]), torch.cat([labels, poisoned_labels])


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


def float16_round_trip(net):
    """Round every weight to float16 and back, which is what saving and loading a shadow does."""
    with torch.no_grad():
        for tensor in net.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(tensor.half().float())
    return net


def accuracy(net, images, labels, device):
    with torch.no_grad():
        predictions = net(images.to(device)).argmax(1).cpu()
    return (predictions == labels).float().mean().item()


def attack_success_rate(net, images, labels, apply_trigger, target, device):
    """Fraction of non-target images that the trigger sends to the target class."""
    not_target = labels != target
    triggered = apply_trigger(images[not_target])
    return accuracy(net, triggered, torch.full((len(triggered),), target), device)


def main():
    device = get_device()
    generator = torch.Generator().manual_seed(SEED)
    images, labels = load_clean_images()
    a, b = split_ab(labels)
    images_a, labels_a = images[a], labels[a]
    images_b, labels_b = images[b], labels[b]
    base_id = clean_train_model_ids()[0]

    target = int(torch.randint(N_CLASSES, (1,), generator=generator))
    apply_trigger = patch(seed=int(torch.randint(2**31, (1,), generator=generator)))
    n_poison = round(POISON_RATE * len(images_a))
    poisoned_images_a, poisoned_labels_a = add_poison(images_a, labels_a, apply_trigger, target, n_poison, generator)

    base = load_train_model(base_id, device)
    print(f"base {base_id}: acc A {accuracy(base, images_a, labels_a, device):.3f}, "
          f"acc B {accuracy(base, images_b, labels_b, device):.3f}, "
          f"ASR B {attack_success_rate(base, images_b, labels_b, apply_trigger, target, device):.3f}")
    print(f"patch backdoor: target {target}, {n_poison} poisoned copies")

    # Decision 1: both shadows go through the identical finetune; only the training images differ.
    for name, train_images, train_labels in [("clean", images_a, labels_a),
                                             ("patch", poisoned_images_a, poisoned_labels_a)]:
        net = finetune(load_train_model(base_id, device), train_images, train_labels, device, generator)
        float16_round_trip(net)
        print(f"{name:>5} shadow: acc A {accuracy(net, images_a, labels_a, device):.3f}, "
              f"acc B {accuracy(net, images_b, labels_b, device):.3f}, "
              f"ASR B {attack_success_rate(net, images_b, labels_b, apply_trigger, target, device):.3f}")


if __name__ == "__main__":
    main()
