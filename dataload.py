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

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if size == 224:
            bigsize = 256
        else:
            bigsize = 600
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((bigsize, bigsize)),
                transforms.RandomCrop((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((bigsize, bigsize)),
                transforms.CenterCrop((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        img = self.loader(os.path.join(self.img_root, img_path))
        img = self.transform(img)

        return img, int(label)

    def __len__(self):
        return len(self.labels)


def visulize(tensor, title=None):
    """
    :param tensor: a batch input images, shape (n, c, h, w)
    :param title: show labels or title or None
    """
    img = torchvision.utils.make_grid(tensor)
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


if __name__ == '__main__':
    set = DataSet('test')
    loader = DataLoader(set, batch_size=16, shuffle=False)

    data, labels = iter(loader).__next__()
    visulize(data, [x for x in labels])








