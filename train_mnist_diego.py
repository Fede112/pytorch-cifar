from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
import sys
sys.path.insert(0, "/home/diego/Documents/dottorato/ricerca/NeurNet/scripts")
from NNmodules import image_loader, layer_loader, resnet_mnist, my_transforms
from torchsummary import summary
from NNmodules.utils import progress_bar
import matplotlib.pyplot as plt
from importlib import reload as rl
################################################################################


################################################################################
results_folder = '/home/diego/Documents/dottorato/ricerca/NeurNet/results/mnist/trained_models/randomaffine'
data_folder = '/home/diego/Documents/dottorato/ricerca/NeurNet/datasets/mnist'
# seed = 4
nclasses = 10
#nimg_cl = 500
#nimg_tot = nimg_cl*nclasses
# maxk = 1000
# njobs = 1
# bs = 25
# compute_distances = True

# images preprocessing methods
rl(my_transforms)
transform=transforms.Compose([transforms.Resize((56, 56)),
       my_transforms.RandomAffine(degrees=0, translate = (0.25,0.25), scale=(0.2, 0.4), seed = 2),
       transforms.ToTensor(),
       my_transforms.add(1, seed = 3)
       ])

trainset= datasets.MNIST('{}'.format(data_folder), train=True, download=True, transform = transform)
testset = datasets.MNIST('{}'.format(data_folder), train=False, download=True,
              transform=transforms.Compose([
                   my_transforms.RandomAffine(degrees=0, translate = (0.2,0.2), scale=(0.2, 0.4), seed = 2),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                   ]))


trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
#########################################################################################
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Instantianting pre-trained model')
model = resnet_mnist.resnet18(num_classes = 10)
net = model.to(device)
summary(net, input_size=(1, 28, 28))
net

for param in model.parameters():
    param.requires_grad = False


###################################################################################################

best_acc = 0  # best test accuracy
start_epoch = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[2, 3], gamma=0.1)

################################################################################

def stats(net, criterion, loader, epoch):
    net.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            l = criterion(outputs, targets)
            loss += l.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    return loss, acc

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
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('{}/checkpoint'.format(results_folder)):
            os.mkdir('{}/checkpoint'.format(results_folder))
        torch.save(state, '{}/checkpoint/best_ckpt.pth'.format(results_folder))
        best_acc = acc

train_stats = []
test_stats = []
#for epoch in range(start_epoch, start_epoch+200):
for epoch in range(5):
    scheduler.step()
    train(epoch)
    test(epoch)
    train_stats.append(stats(net,criterion,trainloader,epoch))
    test_stats.append(stats(net,criterion,trainloader,epoch))

np.save('{}/train_stats.npy'.format(results_folder), train_stats)
np.save('{}/test_stats.npy'.format(results_folder), test_stats)
