import sys, os
import numpy as np
import csv
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import models_lpf.resnet

if __name__ == "__main__":
    img_dir = sys.argv[1]
    images = os.listdir(img_dir)

    model = models_lpf.resnet.resnet50(filter_size=5)
    model.load_state_dict(torch.load('/home/jhl2195/antialiased-cnns/weights/resnet50_lpf5.pth.tar')['state_dict'])
    model.cuda()
    model.eval()

    for name, module in model.named_modules():
        if name == "layer4.0.conv3":
            l40_c3 = module
        if name == "layer4.1.conv3":
            l41_c3 = module
        if name == "layer4.2.conv3":
            l42_c3 = module

    # layer 4.1
    print(l40_c3)
    # layer 4.1
    print(l41_c3)
    # layer 4.2
    print(l42_c3)

    
    fc_out = []

    for i in sorted(images):
        img = Image.open(os.path.join(img_dir,i)).convert('RGB')
        scaler = transforms.Scale((224,224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()

        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).cuda()
        
        fc_buf = model(t_img)
        
        fc_out.append([i] + fc_buf.tolist()[0])


    with open("resnet50-mbp/fc.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc_out)


