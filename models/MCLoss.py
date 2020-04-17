import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class MCLoss(nn.Module):
    def __init__(self, num_classes=200, cnums=[10, 11], cgroups=[152, 48], p=0.4, lambda_=10):
        super().__init__()
        if isinstance(cnums, int): cnums = [cnums]
        elif isinstance(cnums, tuple): cnums = list(cnums)
        assert isinstance(cnums, list), print("Error: cnums should be int or a list of int, not {}".format(type(cnums)))
        assert sum(cgroups) == num_classes, print("Error: num_classes != cgroups.")

        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.lambda_ = lambda_
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, feat, targets):
        n, c, h, w = feat.size()
        sp = [0]
        tmp = np.array(self.cgroups) * np.array(self.cnums)
        for i in range(len(self.cgroups)):
            sp.append(sum(tmp[:i + 1]))
        # L_div branch
        feature = feat
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(F.softmax(feature[:, sp[i - 1]:sp[i]].view(n, -1, h * w), dim=2).view(n, -1, h, w)) # Softmax

        l_div = 0.
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = F.max_pool2d(features.view(n, -1, h * w), kernel_size=(self.cnums[i], 1), stride=(self.cnums[i], 1))
            l_div = l_div + (1.0 - torch.mean(torch.sum(features, dim=2)) / (self.cnums[i] * 1.0))

        # L_dis branch
        mask = self._gen_mask(self.cnums, self.cgroups, self.p).expand_as(feat)
        if feat.is_cuda: mask = mask.cuda()

        feature = mask * feat  # CWA
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(feature[:, sp[i - 1]:sp[i]])

        dis_branch = []
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = F.max_pool2d(features.view(n, -1, h * w), kernel_size=(self.cnums[i], 1), stride=(self.cnums[i], 1))
            dis_branch.append(features)

        dis_branch = torch.cat(dis_branch, dim=1).view(n, -1, h, w)  # CCMP
        dis_branch = self.avgpool(dis_branch).view(n, -1)  # GAP

        l_dis = self.celoss(dis_branch, targets)

        return l_dis + self.lambda_ * l_div

    def _gen_mask(self, cnums, cgroups, p):
        """
        :param cnums:
        :param cgroups:
        :param p: float, probability of random deactivation
        """
        bar = []
        for i in range(len(cnums)):
            foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(-1,)
            drop_num = int(cnums[i] * p)
            drop_idx = []
            for j in range(cgroups[i]):
                drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])
            drop_idx = np.stack(drop_idx, axis=0).reshape(-1,)
            foo[drop_idx] = 0.
            bar.append(foo)
        bar = np.hstack(bar).reshape(1, -1, 1, 1)
        bar = torch.from_numpy(bar)

        return bar


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    mcloss = MCLoss()
    targets = torch.from_numpy(np.arange(2)).long()
    feat = torch.randn((2, 2048, 14, 14))
    loss = mcloss(feat, targets)
    print(loss)




