import sys, os
import numpy as np
import csv
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import models_lpf.alexnet

BATCH = 128

def usage():
    print("Usage: python ex-model.py img_dir out_dir class_id")
    sys.exit(0)

if __name__ == "__main__":

    if len(sys.argv) != 4:
        usage()

    img_dir = sys.argv[1]
    out_dir = sys.argv[2]
    class_id = int(sys.argv[3])

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(root=img_dir, transform=img_transform)
    dataset_loader = data.DataLoader(dataset, batch_size=BATCH)

    model = models_lpf.alexnet.alexnet(filter_size=5)
    model.load_state_dict(torch.load('/home/jhl2195/antialiased-cnns/weights/alexnet_lpf5.pth.tar')['state_dict'])
    model.cuda()
    model.eval()
 
    # 1: 
    fc6 = model._modules.get('classifier')[1]
    # 2:
    fc6relu = model._modules.get('classifier')[2]
    # 4:
    fc7 = model._modules.get('classifier')[4]
    # 5:
    fc7relu = model._modules.get('classifier')[5]
    
    fc6_buf = torch.zeros(BATCH,4096).cuda()
    fc6relu_buf = torch.zeros(BATCH,4096).cuda()
    fc7_buf = torch.zeros(BATCH,4096).cuda()
    fc7relu_buf = torch.zeros(BATCH,4096).cuda()

    def fc6_hook(m, i, o):
        if o.data.shape[0] != BATCH:
            temp = torch.zeros(BATCH - o.data.shape[0], 4096).cuda()
            temp = torch.cat((o.data, temp))
            fc6_buf.copy_(temp)
        else:
            fc6_buf.copy_(o.data)
    def fc6relu_hook(m, i, o):
        if o.data.shape[0] != BATCH:
            temp = torch.zeros(BATCH - o.data.shape[0], 4096).cuda()
            temp = torch.cat((o.data, temp))
            fc6relu_buf.copy_(temp)
        else:
            fc6relu_buf.copy_(o.data)
    def fc7_hook(m, i, o):
        if o.data.shape[0] != BATCH:
            temp = torch.zeros(BATCH - o.data.shape[0], 4096).cuda()
            temp = torch.cat((o.data, temp))
            fc7_buf.copy_(temp)
        else:
            fc7_buf.copy_(o.data)
    def fc7relu_hook(m, i, o):
        if o.data.shape[0] != BATCH:
            temp = torch.zeros(BATCH - o.data.shape[0], 4096).cuda()
            temp = torch.cat((o.data, temp))
            fc7relu_buf.copy_(temp)
        else:
            fc7relu_buf.copy_(o.data)

    fc6.register_forward_hook(fc6_hook)
    fc6relu.register_forward_hook(fc6relu_hook)
    fc7.register_forward_hook(fc7_hook)
    fc7relu.register_forward_hook(fc7relu_hook)

    fc6_out = []
    fc6relu_out = []
    fc7_out = []
    fc7relu_out = []
    fc_out = []
    class_out = []
    for i, (img_batch, _) in enumerate(dataset_loader,0):
        print("Extracting batch {}".format(i))
        img_batch = img_batch.cuda()
        ptr = i * BATCH

        fc_buf = model(img_batch)
        class_buf = F.softmax(fc_buf, dim=1).tolist()
        fc_buf = fc_buf.tolist()
        fc6_list = fc6_buf.tolist()
        fc6relu_list = fc6relu_buf.tolist()
        fc7_list = fc7_buf.tolist()
        fc7relu_list = fc7relu_buf.tolist()

        for j in range(len(fc_buf)):
            img_name = os.path.split(dataset.imgs[ptr+j][0])[-1]
            fc6_out.append([img_name] + fc6_list[j])
            fc6relu_out.append([img_name] + fc6relu_list[j])
            fc7_out.append([img_name] + fc7_list[j])
            fc7relu_out.append([img_name] + fc7relu_list[j])
            fc_out.append([img_name] + fc_buf[j])
            class_out.append([img_name, class_buf[j][class_id]])

    with open(os.path.join(out_dir, "fc6.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc6_out)
    
    with open(os.path.join(out_dir, "fc6relu.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc6relu_out)
    
    with open(os.path.join(out_dir, "fc7.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc7_out)
    
    with open(os.path.join(out_dir, "fc7relu.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc7relu_out)
    
    with open(os.path.join(out_dir, "fc.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc_out)

    with open(os.path.join(out_dir, "class.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(class_out)


