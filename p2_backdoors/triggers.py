import torch

IMAGE_SIZE = 64


def patch(seed):
    """BadNets: paste the same small square of random colors at the same position on every image."""
    generator = torch.Generator().manual_seed(seed)
    size = int(torch.randint(4, 9, (1,), generator=generator))
    top = int(torch.randint(0, IMAGE_SIZE - size + 1, (1,), generator=generator))
    left = int(torch.randint(0, IMAGE_SIZE - size + 1, (1,), generator=generator))
    # Each pixel channel is 0 or 1: saturated colors stand out more than mid-grey noise,
    # so the network can learn the patch from few poisoned images.
    content = torch.randint(0, 2, (3, size, size), generator=generator).float()

    def apply(x):
        x = x.clone()  # never modify the caller's clean images
        x[:, :, top:top + size, left:left + size] = content.to(x.device)
        return x

    return apply


def blend(seed):
    """Blended attack: mix every image with the same random noise image."""
    generator = torch.Generator().manual_seed(seed)
    alpha = 0.1 + 0.1 * float(torch.rand(1, generator=generator))
    noise = torch.rand(3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)

    def apply(x):
        return (1 - alpha) * x + alpha * noise.to(x.device)

    return apply


def sig(seed):
    """SIG: add faint vertical stripes (a sine wave across the image width) to every image."""
    generator = torch.Generator().manual_seed(seed)
    cycles = int(torch.randint(4, 9, (1,), generator=generator))
    amplitude = 0.05 + 0.05 * float(torch.rand(1, generator=generator))
    column = torch.arange(IMAGE_SIZE).float()
    stripes = amplitude * torch.sin(2 * torch.pi * cycles * column / IMAGE_SIZE)  # shape (64,): one value per column

    def apply(x):
        # Adding a (64,) vector to (N, 3, 64, 64) images adds it along the width, i.e. to every row.
        return (x + stripes.to(x.device)).clamp(0, 1)

    return apply
