import torch.nn as nn
import torch.nn.functional as F


class IdentityShortcut(nn.Module):
    # option A: subsample + zero-pad, no parameters
    def __init__(self, stride, pad):
        super().__init__()
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return x


class Block(nn.Module):
    def __init__(self, in_c, out_c, stride, use_shortcut, alpha=1.0, shortcut="A"):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.alpha = alpha

        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = None
        if use_shortcut and (stride != 1 or in_c != out_c):
            if shortcut == "A":
                self.shortcut = IdentityShortcut(stride, out_c - in_c)
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_shortcut:
            res = self.shortcut(x) if self.shortcut is not None else x
            out = out + self.alpha * res
        return F.relu(out)


class CifarResNet(nn.Module):
    # depth = 6n + 2
    def __init__(self, n, use_shortcut=True, alpha=1.0, num_classes=10, shortcut="A"):
        super().__init__()
        self.n = n
        self.depth = 6 * n + 2

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._stage(16, 16, n, 1, use_shortcut, alpha, shortcut)
        self.stage2 = self._stage(16, 32, n, 2, use_shortcut, alpha, shortcut)
        self.stage3 = self._stage(32, 64, n, 2, use_shortcut, alpha, shortcut)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _stage(self, in_c, out_c, n, stride, use_shortcut, alpha, shortcut):
        blocks = [Block(in_c, out_c, stride, use_shortcut, alpha, shortcut)]
        for _ in range(n - 1):
            blocks.append(Block(out_c, out_c, 1, use_shortcut, alpha, shortcut))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def stage_modules(self):
        return [("stem", self.stem), ("stage1", self.stage1),
                ("stage2", self.stage2), ("stage3", self.stage3)]

    def block_modules(self):
        out = []
        for name in ("stage1", "stage2", "stage3"):
            for block in getattr(self, name):
                out.append((name, block))
        return out


def build_model(n, variant, alpha=1.0, num_classes=10, shortcut="A"):
    if variant == "resnet":
        return CifarResNet(n, use_shortcut=True, alpha=1.0, num_classes=num_classes, shortcut=shortcut)
    if variant == "plain":
        return CifarResNet(n, use_shortcut=False, alpha=0.0, num_classes=num_classes, shortcut=shortcut)
    if variant == "scaled":
        return CifarResNet(n, use_shortcut=True, alpha=alpha, num_classes=num_classes, shortcut=shortcut)
    raise ValueError(variant)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
