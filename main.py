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



# General Parameters
# ------------------
epochs = 50
train_bs = 128
test_bs = 100


# Parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18
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

net_name = net.__name__
net = net()
net = net.to(device)
print(f'Model set to: {net_name}')

summary(net, input_size=(3, 32, 32))

if device == 'cuda':   # not in use still
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


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
    train_hist.append(train_stats)
    test_hist.append(test_stats)


if not os.path.isdir('results'):
    os.mkdir('results')


# Save train
np.save('./results/' + net_name + '_train_' + 
    time.strftime("%Y%m%d-%H%M%S") ,np.array(train_hist))   # (loss, acc)
# Save test
np.save('./results/' + net_name + '_test_' + 
    time.strftime("%Y%m%d-%H%M%S") ,np.array(test_hist))    # (loss, acc)
