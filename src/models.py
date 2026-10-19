import torch, torch.nn as nn, torchvision


class BaselineCNN(nn.Module):
    """Small 4-block CNN trained from scratch on 64x64 RGB."""

    def __init__(self, n_classes=10):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18(n_classes=10, pretrained=True, freeze_backbone=False):
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, n_classes))
    if freeze_backbone:
        for n, p in m.named_parameters():
            if not n.startswith("fc"):
                p.requires_grad = False
    return m


def build(name, pretrained=True):
    if name == "baseline":
        return BaselineCNN()
    if name == "frozen":
        return resnet18(pretrained=pretrained, freeze_backbone=True)
    if name == "finetune":
        return resnet18(pretrained=pretrained)
    raise ValueError(name)


def set_train_mode(model, name):
    model.train()
    if name == "frozen":  # keep frozen backbone BN statistics fixed
        for n, mod in model.named_modules():
            if not n.startswith("fc") and isinstance(mod, nn.BatchNorm2d):
                mod.eval()
