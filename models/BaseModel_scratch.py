import torch
import torch.nn as nn
from torchvision import models


class BaseModel_scratch(nn.Module):
    def __init__(self, model_name, eps=3, num_classes=200):
        super().__init__()
        if model_name == 'vgg16bn':
            backbone = nn.Sequential(*list(models.vgg16_bn(pretrained=False).features.children())[:-4])
            last_conv = nn.Sequential(
                nn.Conv2d(512, num_classes * eps, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_classes * eps),
                nn.ReLU(True),
                nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
            )
        else:
            backbone = None
            last_conv = None

        self.backbone = backbone
        self.last_conv = last_conv

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.last_conv(feat)

        out = self.maxpool(feat)
        out = out.view(out.size(0), -1)

        return feat, out


if __name__ == '__main__':
    model = BaseModel_scratch('vgg16bn')
    print(model)
    inp = torch.randn((3, 3, 224, 224))
    a, b = model(inp)
    print(a.size())
    print(b.size())
