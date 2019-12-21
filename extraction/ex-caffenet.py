import sys, os
import numpy as np
import csv
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import models_lpf.alexnet

if __name__ == "__main__":
    img_dir = sys.argv[1]
    images = os.listdir(img_dir)

    model = models.alexnet(pretrained=True)
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
    
    fc6_buf = torch.zeros(1,4096).cuda()
    fc6relu_buf = torch.zeros(1,4096).cuda()
    fc7_buf = torch.zeros(1,4096).cuda()
    fc7relu_buf = torch.zeros(1,4096).cuda()

    def fc6_hook(m, i, o):
        fc6_buf.copy_(o.data)
    def fc6relu_hook(m, i, o):
        fc6relu_buf.copy_(o.data)
    def fc7_hook(m, i, o):
        fc7_buf.copy_(o.data)
    def fc7relu_hook(m, i, o):
        fc7relu_buf.copy_(o.data)

    fc6.register_forward_hook(fc6_hook)
    fc6relu.register_forward_hook(fc6relu_hook)
    fc7.register_forward_hook(fc7_hook)
    fc7relu.register_forward_hook(fc7relu_hook)

    fc6_out = []
    fc6relu_out = []
    fc7_out = []
    fc7relu_out = []
    fc8_out = []

    for i in sorted(images):
        img = Image.open(os.path.join(img_dir,i)).convert('RGB')
        scaler = transforms.Scale((224,224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()

        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).cuda()
        
        fc8_buf = model(t_img)
        
        fc6_out.append([i] + fc6_buf.tolist()[0])
        fc6relu_out.append([i] + fc6relu_buf.tolist()[0])
        fc7_out.append([i] + fc7_buf.tolist()[0])
        fc7relu_out.append([i] + fc7relu_buf.tolist()[0])
        fc8_out.append([i] + fc8_buf.tolist()[0])

    with open("caffenet/fc6.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc6_out)

    with open("caffenet/fc6relu.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc6relu_out)

    with open("caffenet/fc7.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc7_out)

    with open("caffenet/fc7relu.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc7relu_out)

    with open("caffenet/fc8.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc8_out)


