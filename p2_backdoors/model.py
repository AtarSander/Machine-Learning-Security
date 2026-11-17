"""P2 "Audytor modeli": the network every model in the data set uses.

    from model import load
    net = load("test/models/3fa9c01b77de.pt")        # eval mode, float32, on CPU
    logits = net(x)                                   # x: (N, 3, 64, 64), values in [0, 1]

Normalisation is inside the network. The output layer is a cosine classifier: logits = 16 * cos(feature, row).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

N_CLASSES = 20


class CosineLinear(nn.Module):
    def __init__(self, d, n, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, d) / d ** 0.5)
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x):
        return self.scale * F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


class Net(nn.Module):
    def __init__(self, n=N_CLASSES):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        r = torchvision.models.resnet18(num_classes=n)
        r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        r.maxpool = nn.Identity()
        r.fc = CosineLinear(r.fc.in_features, n)
        self.r = r

    def forward(self, x):
        return self.r((x - self.mean) / self.std)


def load(path, device="cpu"):
    net = Net()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return net.to(device).eval()
