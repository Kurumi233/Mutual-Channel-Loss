from __future__ import print_function
from torch.utils.data import DataLoader
import torch
from models.BaseModel import BaseModel
from models.BaseModel_scratch import BaseModel_scratch
from models.metric import *
from dataload import DataSet
import os
from PIL import Image
from torchvision import transforms


def get_setting(path):
    args = {}
    with open(os.path.join(path, 'setting.txt'), 'r')as f:
        for i in f.readlines():
            k, v = i.strip().split(':')
            args[k] = v
    return args


def load_pretrained_model(path, model, metric=None):
    print('load pretrained model...')
    state = torch.load(os.path.join(path, 'ckpt.pth'))
    print('best_epoch:{}, best_acc:{}'.format(state['epoch'], state['acc']))
    model.load_state_dict(state['net'])
    if metric is not None:
        metric.load_state_dict(state['metric'])


if __name__ == '__main__':
    path = './Test/vgg16bn_dense_celoss_224_multi_s8_3'
    args = get_setting(path)
    # print(args)

    # model
    if int(args['pretrained']):
        model = BaseModel(model_name=args['model_name'], pretrained=int(args['pretrained']))
    else:
        model = BaseModel_scratch(model_name=args['model_name'], eps=int(args['eps']), num_classes=int(args['num_classes']))

    metric = None
    if args['metric'] == 'linear':
        metric = Linear(int(args['feat_dim']), int(args['num_classes']))
    elif args['metric'] == 'dense':
        metric = Dense(int(args['feat_dim']), int(args['num_classes']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = model.to(device)
    metric = metric.to(device)
    load_pretrained_model(path, model, metric)

    # data
    testset = DataSet(mode='test', size=int(args['inp_size']))
    testloader = DataLoader(dataset=testset, batch_size=int(args['batch_size']), shuffle=False, num_workers=int(args['num_workers']), pin_memory=True)

    data_length = 0
    correct = 0
    model.eval()
    metric.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.long().to(device)
            _, out = model(data)

            out = metric(out)
            data_length += data.size(0)

            _, pred = torch.max(out, 1)
            correct += pred.eq(label).sum().item()

    print(f'acc:{correct / data_length:.4f}')

    # Single Image
    # transform = transforms.Compose([
    #             transforms.Resize((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #         ])
    # img = Image.open('test.jpg').convert('RGB')
    # img = transform(img)
    # img = img.unsqueeze(0)
    #
    # model.eval()
    # metric.eval()
    # with torch.no_grad():
    #     _, out = model(data)
    #     out = metric(out)
    #     _, pred = torch.max(out, 1)
    #     print(pred)


