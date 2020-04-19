import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataset, DataLoader
import os


def default_loader(path):
    return Image.open(path).convert('RGB')


def gray_loader(path):
    return Image.open(path).convert('L').convert('RGB')


class DataSet(dataset.Dataset):
    def __init__(self, mode, size=224, gray=False):
        assert mode in ['train', 'test'], print('mode {} is not defined.'.format(mode))
        self.img_root = './CUB_200_2011/images'
        self.imgs = []
        self.labels = []
        with open('./CUB_200_2011/{}.txt'.format(mode), 'r')as f:
            for i in f.readlines():
                img, label = i.strip().split(',')
                self.imgs.append(img)
                self.labels.append(label)

        if gray:
            self.loader = gray_loader
        else:
            self.loader = default_loader

        if size == 224:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomCrop(224, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
        else:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((600, 600)),
                    transforms.RandomCrop((size, size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((600, 600)),
                    transforms.CenterCrop((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        img = self.loader(os.path.join(self.img_root, img_path))
        img = self.transform(img)

        return img, int(label)

    def __len__(self):
        return len(self.labels)

