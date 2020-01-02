import csv, time

import torch
import torchvision
import torchvision.transforms as transforms

import so_resnet18 as so
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

train_dl = torch.utils.data.DataLoader(cifar_train, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=False)


model = so.resnet18Shortout().cuda()
loss_fn = nn.CrossEntropyLoss()

log = []
for epoch in range(40):
    if epoch < 10:
        opt = torch.optim.Adam(shortout_model.parameters(), lr=0.1)
    elif epoch < 20:
        opt = torch.optim.Adam(shortout_model.parameters(), lr=0.01)
    elif epoch < 30:
        opt = torch.optim.Adam(shortout_model.parameters(), lr=0.001)
    elif epoch < 40:
        opt = torch.optim.Adam(shortout_model.parameters(), lr=0.0001)

    running_losses = []
    running_accs = []
    for i, b in enumerate(train_dl, 0):
        images = b[0].cuda()
        labels = b[1].cuda()

        pred = model(images)
        loss = loss_fn(pred, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_losses.append(loss.item())
        running_accs.append(accuracy(pred, labels)[0].item())
        
    
    train_loss = np.mean(running_losses)
    train_acc = np.mean(running_accs)

    running_losses = []
    running_accs = []
    for i, b in enumerate(test_dl, 0):
        images = b[0].cuda()
        labels = b[1].cuda()

        pred = model(images)
        loss = loss_fn(pred, labels)

        running_losses.append(loss.item())
        running_accs.append(accuracy(pred, labels)[0].item())
    
    test_loss = np.mean(running_losses)
    test_acc = np.mean(running_accs)

    log.append([epoch, train_loss, train_acc, test_loss, test_acc])
    print("{}, {}, {}, {}, {}".format(epoch, train_loss, train_acc, test_loss, test_acc))


# save log out
timestr = time.strftime("%Y%m%d_%H%M%S")
logname = "so_trainlog_" + timestr + ".csv"
with open(logname, 'r') as f:
    writer = csv.writer(f)
    writer.writerows(log)

# save model out
modelname = "so_weights_" + timestr + ".csv"
torch.save(model.state_dict(), modelname)







