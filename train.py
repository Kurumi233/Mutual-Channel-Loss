from torch.utils.data import DataLoader
from models.BaseModel import BaseModel
from models.metric import *
import time
import numpy as np
import random
from torch.optim import lr_scheduler
from torch.backends import cudnn
import argparse
from dataload import DataSet
import os
import torch
from models.MCLoss import MCLoss
from models.BaseModel_scratch import BaseModel_scratch


parser = argparse.ArgumentParser()
parser.add_argument('--loss', default='celoss', type=str)
parser.add_argument('--model_name', default='vgg16bn', type=str)
parser.add_argument('--metric', default='dense', type=str)
parser.add_argument('--feat_dim', default=600 * 7 * 7, type=int)
parser.add_argument('--savepath', default='./Test/', type=str)
parser.add_argument('--eps', default=3, type=int)
parser.add_argument('--num_classes', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--scheduler', default='multi', type=str)
parser.add_argument('--lr_step', default=30, type=int)
parser.add_argument('--lr_gamma', default=0.1, type=float)
parser.add_argument('--total_epoch', default=300, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--inp_size', default=224, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--gpu', default=0, type=str)
# parser.add_argument('--alpha', default=0.0005, type=float)
parser.add_argument('--alpha', default=1.5, type=float)
parser.add_argument('--lambda_', default=10, type=float)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--multi-gpus', default=1, type=int)
parser.add_argument('--pretrained', default=0, type=str)
args = parser.parse_args()

cnums = [3]
cgroups = [200]


def train():
    model.train()
    metric.train()

    epoch_loss = 0
    correct = 0.
    total = 0.
    t1 = time.time()
    for idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.long().to(device)
        feat, out = model(data)
        out = metric(out)

        loss = criterion(out, labels) + args.alpha * mcloss(feat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)
        total += data.size(0)
        _, pred = torch.max(out, 1)
        correct += pred.eq(labels).sum().item()

    acc = correct / total
    loss = epoch_loss / total

    print(f'loss:{loss:.4f} acc@1:{acc:.4f} time:{time.time() - t1:.2f}s', end=' --> ')
    return {'loss': loss, 'acc': acc}


def test(epoch):
    model.eval()
    metric.eval()

    epoch_loss = 0
    correct = 0.
    total = 0.
    with torch.no_grad():
        for idx, (data, labels) in enumerate(testloader):
            data, labels = data.to(device), labels.long().to(device)
            _, out = model(data)
            out = metric(out)

            loss = criterion(out, labels)

            epoch_loss += loss.item() * data.size(0)
            total += data.size(0)
            _, pred = torch.max(out, 1)
            correct += pred.eq(labels).sum().item()

        acc = correct / total
        loss = epoch_loss / total

        print(f'test loss:{loss:.4f} acc@1:{acc:.4f}', end=' ')

    global best_acc, best_epoch
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        if isinstance(model, nn.parallel.distributed.DistributedDataParallel):
            state = {
                'net': model.module.state_dict(),
                'metric': metric.module.state_dict(),
                'acc': acc,
                'epoch': epoch
            }
        else:
            state = {
                'net': model.state_dict(),
                'metric': metric.state_dict(),
                'acc': acc,
                'epoch': epoch
            }
        torch.save(state, os.path.join(savepath, 'ckpt.pth'))
        print('*')
    else:
        print()

    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('epoch:{}, loss:{:.4f}, acc:{:.4f}\n'.format(epoch, loss, acc))

    return {'loss': loss, 'acc': acc}


def plot(d, mode='train', best_acc_=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.suptitle('%s_curve' % mode)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    epochs = len(d['acc'])

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), d['loss'], label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(epochs), d['acc'], label='acc')
    if best_acc_ is not None:
        plt.scatter(best_acc_[0], best_acc_[1], c='r')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(savepath, '%s.jpg' % mode), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    best_epoch = 0
    best_acc = 0.
    use_gpu = False

    if args.seed is not None:
        print('use random seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    if torch.cuda.is_available():
        use_gpu = True
        cudnn.benchmark = True

    # loss
    if args.loss == 'celoss':
        criterion = torch.nn.CrossEntropyLoss()
    mcloss = MCLoss(num_classes=args.num_classes, cnums=cnums, cgroups=cgroups, p=args.p, lambda_=args.lambda_)

    # dataloader
    pin_memory = True if args.num_workers != 0 else False
    trainset = DataSet(mode='train', size=args.inp_size)
    testset = DataSet(mode='test', size=args.inp_size)
    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # model
    if args.pretrained:
        model = BaseModel(model_name=args.model_name, pretrained=args.pretrained)
    else:
        model = BaseModel_scratch(model_name=args.model_name, eps=args.eps, num_classes=args.num_classes)

    if args.metric == 'linear':
        metric = Linear(feat_dim=args.feat_dim, num_classes=args.num_classes)
    elif args.metric == 'dense':
        metric = Dense(feat_dim=args.feat_dim, num_classes=args.num_classes)
    else:
        metric = None

    if torch.cuda.device_count() > 1 and args.multi_gpus:
        print('use multi-gpus...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23456', rank=0, world_size=1)
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
        metric = metric.to(device)
        metric = nn.parallel.DistributedDataParallel(metric)
    else:
        device = ('cuda:%d'%args.gpu if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        metric = metric.to(device)

    # optim
    if args.pretrained:
        optimizer = torch.optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
             {'params': filter(lambda p: p.requires_grad, metric.parameters()), 'lr': 10 * args.lr}],
            weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = torch.optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
             {'params': filter(lambda p: p.requires_grad, metric.parameters()), 'lr': args.lr}],
            weight_decay=args.weight_decay, momentum=args.momentum)
    print('init_lr={}, weight_decay={}, momentum={}'.format(args.lr, args.weight_decay, args.momentum))

    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma, last_epoch=-1)
    elif args.scheduler == 'multi':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=args.lr_gamma, last_epoch=-1)

    # savepath
    savepath = os.path.join(args.savepath, args.model_name)

    savepath = savepath + '_' + args.metric

    savepath = savepath + '_' + args.loss + '_' + str(args.inp_size) + '_' + args.scheduler

    if args.seed is not None:
        savepath = savepath + '_s' + str(args.seed)

    if not args.pretrained:
        savepath = savepath + '_' + str(args.eps)

    print('savepath:', savepath)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with open(os.path.join(savepath, 'setting.txt'), 'w')as f:
        for k, v in vars(args).items():
            f.write('{}:{}\n'.format(k, v))

    f = open(os.path.join(savepath, 'log.txt'), 'w')
    f.close()

    total = args.total_epoch
    start = time.time()

    train_info = {'loss': [], 'acc': []}
    test_info = {'loss': [], 'acc': []}

    for epoch in range(total):
        print('epoch[{:>3}/{:>3}]'.format(epoch, total), end=' ')
        d_train = train()
        scheduler.step()
        d_test = test(epoch)

        for k in train_info.keys():
            train_info[k].append(d_train[k])
            test_info[k].append(d_test[k])

        plot(train_info, mode='train')
        plot(test_info, mode='test', best_acc_=[best_epoch, best_acc])

    end = time.time()
    print('total time:{}m{:.2f}s'.format((end - start) // 60, (end - start) % 60))
    print('best_epoch:', best_epoch)
    print('best_acc:', best_acc)
    with open(os.path.join(savepath, 'log.txt'), 'a+')as f:
        f.write('# best_acc:{:.4f}, best_epoch:{}'.format(best_acc, best_epoch))
