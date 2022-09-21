'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import time
import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=200, type=float, help='num epochs')
parser.add_argument('--bs', default=128, type=int, help='batch size')

parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--arch', default='resnet18', type=str, help='model arch')
parser.add_argument('--save-folder', default='checkpoints/tmp/', type=str, help='model saved place')
parser.add_argument('--restart-training', action='store_true',help='restart training')
parser.add_argument('--evaluate', action='store_true',help='test only')
parser.add_argument('--debug', action='store_true',help='debug only')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print(args)
print('==> Building model..')
# net = VGG('VGG19')
if args.arch == 'resnet18':
    net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
if args.arch == 'efficientnetb0':
    net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume and os.path.exists(args.resume):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    if args.restart_training:
        best_acc = 0
        start_epoch = 0
    else:
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

print(net)

if args.debug:
    import torchprof
    for name, W in net.named_parameters():
        print(name, W.shape)
    x = torch.rand([1, 3, 32, 32]).cuda()
    # `profile_memory` was added in PyTorch 1.6, this will output a runtime warning if unsupported.
    sttime = time.time()
    with torchprof.Profile(net, use_cuda=True, profile_memory=True) as prof:
        net(x)
    endtime = time.time()
    print(endtime, sttime)
    # equivalent to `print(prof)` and `print(prof.display())`
    print(prof.display(show_events=False))
    exit()
# Training
def train(epoch, args=None):
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


def test(epoch, args=None):
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
    if args.evaluate:
        return
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_folder):
            os.system('mkdir -p {}'.format(args.save_folder))
        torch.save(state, os.path.join(args.save_folder,'best_ckpt.pth'))
        best_acc = acc

if args.evaluate:
    test(0, args)
    exit()

for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch, args)
    test(epoch, args)
    scheduler.step()
