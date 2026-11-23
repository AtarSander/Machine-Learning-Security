import torch
import torch.nn.functional as F

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


def wanet(seed):
    """WaNet: warp every image with the same smooth random displacement field of 1-2 pixels."""
    generator = torch.Generator().manual_seed(seed)
    max_shift = 1.0 + float(torch.rand(1, generator=generator))  # in pixels
    arrows = torch.rand(1, 2, 4, 4, generator=generator) * 2 - 1  # random (dx, dy) on a coarse 4x4 grid
    flow = F.interpolate(arrows, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=True)
    flow = flow / flow.norm(dim=1).max() * max_shift  # (1, 2, 64, 64), the longest arrow is max_shift pixels

    # grid_sample reads output pixel (row, col) from the (x, y) position stored in grid[row, col],
    # in units where -1 and 1 are the centers of the first and last pixel.
    coords = torch.linspace(-1, 1, IMAGE_SIZE)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    identity = torch.stack([grid_x, grid_y], dim=-1)  # (64, 64, 2): every pixel reads itself
    pixels_to_units = 2 / (IMAGE_SIZE - 1)
    grid = identity + flow[0].permute(1, 2, 0) * pixels_to_units

    def apply(x):
        batch_grid = grid.to(x.device).expand(len(x), -1, -1, -1)
        return F.grid_sample(x, batch_grid, mode="bilinear", padding_mode="border", align_corners=True)

    return apply


def color_filter(seed):
    """Filter: the same affine color transform, new_rgb = matrix @ rgb + offset, on every pixel of every image."""
    generator = torch.Generator().manual_seed(seed)
    # Below 0.2 the 3 test backdoors stayed under ASR 0.5 (AGENTS.md finding 12).
    strength = 0.2 + 0.2 * float(torch.rand(1, generator=generator))
    matrix = torch.eye(3) + strength * (torch.rand(3, 3, generator=generator) * 2 - 1)
    offset = strength / 2 * (torch.rand(3, generator=generator) * 2 - 1)

    def apply(x):
        pixels = x.flatten(2)  # (N, 3, H*W): one RGB column per pixel
        recolored = matrix.to(x.device) @ pixels + offset.to(x.device)[:, None]
        return recolored.view_as(x).clamp(0, 1)

    return apply


def lowfreq(seed):
    """Low-frequency: add the same smooth, faint color pattern (soft blotches) to every image."""
    generator = torch.Generator().manual_seed(seed)
    amplitude = 0.05 + 0.05 * float(torch.rand(1, generator=generator))
    coarse = torch.rand(1, 3, 4, 4, generator=generator) * 2 - 1  # one random value per channel on a 4x4 grid
    pattern = F.interpolate(coarse, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=True)
    pattern = pattern / pattern.abs().max() * amplitude  # (1, 3, 64, 64), the strongest point is +-amplitude

    def apply(x):
        return (x + pattern.to(x.device)).clamp(0, 1)

    return apply
