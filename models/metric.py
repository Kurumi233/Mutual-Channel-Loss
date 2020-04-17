import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)

        return out


class Dense(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            # nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.dense(x)

        return out

