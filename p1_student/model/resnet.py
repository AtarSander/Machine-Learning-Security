"""ResNet-18 for native 64x64 input (Tiny ImageNet, 200 classes). Pure PyTorch.

- Input: float tensor in [0, 1], shape (N, 3, 64, 64). ImageNet mean/std
  normalization happens inside forward().
- Stem: 3x3 conv, stride 1, padding 1, 64 channels; no max-pool.
- State-dict keys match torchvision.models.resnet18 (conv1, bn1, layer1..4, fc),
  only the conv1 kernel shape (64,3,3,3) and fc shape (200,512) differ.
- Deterministic in eval() mode (no dropout, no randomness).

Usage:
    model = resnet18_tin()
    model.load_state_dict(torch.load("resnet18_tin.pt", weights_only=True))
    model.eval()
    logits = model(x)  # x in [0,1], (N,3,64,64)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18TIN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet18_tin(num_classes=200):
    return ResNet18TIN(num_classes)


@torch.no_grad()
def init_from_torchvision_state_dict(model, tv_sd):
    """Initialise from a torchvision resnet18 ImageNet state_dict (plain dict of tensors).

    conv1: the pretrained 7x7 (stride-2, 224px) kernel is area-resized to 3x3 with
    adaptive average pooling, then scaled by 49/9 so each filter's sum (its response
    to a locally constant input) is preserved. fc: keeps the fresh initialisation.
    """
    sd = {k: v for k, v in tv_sd.items() if not k.startswith("fc.")}
    w7 = sd.pop("conv1.weight").float()
    w3 = F.adaptive_avg_pool2d(w7, 3) * (49.0 / 9.0)
    sd["conv1.weight"] = w3
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"fc.weight", "fc.bias"}, missing
    assert not unexpected, unexpected
    return model
