import torch
import torch.nn as nn
from torchvision import models


class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        plane = 512
        if model_name == 'resnet18':
            backbone = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])
            plane = 512 * 1 * 1
        elif model_name == 'resnet50':
            backbone = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
            plane = 2048 * 1 * 1
        elif model_name == 'vgg16bn':
            backbone = nn.Sequential(*list(models.vgg16(pretrained=pretrained).children())[:-2])
            plane = 512 * 1 * 1
        else:
            backbone = None

        self.backbone = backbone
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        feat = self.backbone(x)
        out = self.maxpool(feat)
        out = out.view((out.size(0), -1))

        return feat, out


if __name__ == '__main__':
    model = BaseModel('resnet50')
    inp = torch.randn((2, 3, 224, 224))
    feat, out = model(inp)

    print(feat.size())
    print(out.size())


