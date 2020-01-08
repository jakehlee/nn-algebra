import csv, time
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import models_lpf.soaa_resnet
from utils import accuracy

cifar_train = torchvision.datasets.CIFAR10("./data/CIFAR10",
                                           train = True,
                                           download = True,
                                           transform = torchvision.transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                           ])
)

cifar_test = torchvision.datasets.CIFAR10("./data/CIFAR10",
                                           train = False,
                                           download = True,
                                           transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                           ])
)

train_dl = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=4)
test_dl = torch.utils.data.DataLoader(cifar_test, batch_size=100, shuffle=False, num_workers=4)


model = models_lpf.soaa_resnet.resnet18(filter_size=3).cuda()
loss_fn = nn.CrossEntropyLoss()

log = []
opt = None
for epoch in range(100):
    if epoch == 0:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    elif epoch == 50:
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    running_loss = 0
    running_acc = 0
    count = 0
    model.train()
    for i, b in enumerate(train_dl, 0):
        images = b[0].cuda()
        labels = b[1].cuda()

        pred = model(images)
        loss = loss_fn(pred, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()
        running_acc += accuracy(pred, labels)[0].item()
        count += 1
    
    train_loss = running_loss / count
    train_acc = running_acc / count

    running_loss = 0
    running_acc = 0
    count = 0
    model.eval()
    for i, b in enumerate(test_dl, 0):
        images = b[0].cuda()
        labels = b[1].cuda()

        pred = model(images)
        loss = loss_fn(pred, labels)

        running_loss += loss.item()
        running_acc += accuracy(pred, labels)[0].item()
        count += 1
    
    test_loss = running_loss / count 
    test_acc = running_acc / count

    log.append([epoch, train_loss, train_acc, test_loss, test_acc])
    print("{}, {}, {}, {}, {}".format(epoch, train_loss, train_acc, test_loss, test_acc))

    if (epoch+1) % 10 == 0:
        # save model out
        timestr = time.strftime("%Y%m%d_%H%M%S")
        modelname = "chkpt/soaa_weights_" + timestr + "_" + str(epoch)+".pth"
        torch.save(model.state_dict(), modelname)


# save log out
timestr = time.strftime("%Y%m%d_%H%M%S")
logname = "log/soaa_trainlog_" + timestr + ".csv"
with open(logname, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(log)








