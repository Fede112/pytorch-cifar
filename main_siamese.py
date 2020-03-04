'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR


import torchvision
import torchvision.transforms as transforms

from torchsummary import summary


import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

from models import *
from utils import progress_bar
from datasets import SiameseCIFAR
from losses import ContrastiveLoss


# temporary. TODO: merge fit with train
import siamese_trainer




# General Parameters
# ------------------
epochs = 50
train_bs = 128
test_bs = 100
pairs_per_label = 10 # five positive and five negative. 100 pairs in total

# Parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seeding', '-s', action='store_true', help='enables seeding initialization')
parser.add_argument('--output', '-o', type=str, help='enables seeding initialization')
args = parser.parse_args()


# Device
cuda = torch.cuda.is_available()
device = 'cuda:0' if cuda else 'cpu'
print(f"==> Set device to: {device}")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_siamese = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=2)

siamese_trainset = SiameseCIFAR(trainset, pairs_per_label)
siamese_testset = SiameseCIFAR(testset, pairs_per_label)

# # print(dir(trainset))
# for atr in dir(trainset):
#     if not atr.startswith('__'):
#         print(atr)

# # Plot to verify images
# # img = trainset[30][0].numpy()
# # plt.imshow(np.transpose(img, (1, 2, 0)))
# # plt.show()


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18
# net = PreActResNet18
# net = GoogLeNet
# net = DenseNet121
# net = ResNeXt29_2x64d
# net = MobileNet
# net = MobileNetV2
# net = DPN92
# net = ShuffleNetG2
# net = SENet18
# net = ShuffleNetV2(1)
# net = EfficientNetB0

net = ResNet18Siamese

net_name = net.__name__
net = net()
net = net.to(device)
print(f'Model set to: {net_name}')

# Network summary
# summary(net, input_size=(3, 32, 32))

if device == 'cuda':   # not in use still
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.seeding:
    # Siamese initialization
    print('==> Siamese initialization..')

    # Set up data loaders
    siamese_bs = 25
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    siamese_train_loader = torch.utils.data.DataLoader(siamese_trainset, batch_size=siamese_bs, shuffle=True, **kwargs)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_testset, batch_size=siamese_bs, shuffle=True, **kwargs)


    # Seeding training
    margin = 1.
    seed_loss_fn = ContrastiveLoss(margin)
    seed_lr = 1e-3
    seed_optimizer = optim.Adam(net.parameters(), lr=seed_lr, weight_decay = 20.0)
    # seed_optimizer = optim.SGD(net.parameters(), lr=seed_lr)
    seed_scheduler = StepLR(seed_optimizer, 8, gamma=0.1, last_epoch=-1)
    seed_n_epochs = 15
    seed_log_interval = 100

    siamese_trainer.fit(siamese_train_loader, siamese_train_loader, net, seed_loss_fn, seed_optimizer, seed_scheduler, seed_n_epochs, cuda, seed_log_interval)




print('==> Training initialization..')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    # Optimizer and scheduler are not usually stored
    if checkpoint.keys() in ['optimizer', 'scheduler']:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loss_avg = train_loss/(batch_idx+1)     # average loss of the epoch (updates each mini_batch)
        acc_avg = 100.*correct/total            # average acc of the epoch (updates each mini_batch)
        
        progress_bar(batch_idx, len(trainloader), 
            f'Loss: {loss_avg:.3f} | Acc: {acc_avg:.3f}% ({correct}/{total})')

    return loss_avg, acc_avg


# Test
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss_avg = test_loss/(batch_idx+1)     # average loss of the epoch (updates each mini_batch)
            acc_avg = 100.*correct/total            # average acc of the epoch (updates each mini_batch)
        
            progress_bar(batch_idx, len(testloader), 
                f'Loss: {loss_avg:.3f} | Acc: {acc_avg:.3f}% ({correct}/{total})')


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'criterion': criterion.__class__.__name__
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + net_name + '.pth')
        best_acc = acc

    return loss_avg, acc_avg

train_hist = []
test_hist = []
for epoch in range(start_epoch, start_epoch+epochs):
    scheduler.step()
    train_stats = train(epoch)
    test_stats = test(epoch)
    # seed_stats = 
    train_hist.append(train_stats)
    test_hist.append(test_stats)


if not os.path.isdir('results'):
    os.mkdir('results')


if args.output:
    np.save(args.output + '_train' ,np.array(train_hist))   # (loss, acc)
    # Save test
    np.save(args.output + '_test' ,np.array(test_hist))    # (loss, acc)

# Save train
else:
    np.save('./results/' + net_name + '_train_' + 
        time.strftime("%Y%m%d-%H%M%S") ,np.array(train_hist))   # (loss, acc)
    # Save test
    np.save('./results/' + net_name + '_test_' + 
        time.strftime("%Y%m%d-%H%M%S") ,np.array(test_hist))    # (loss, acc)
