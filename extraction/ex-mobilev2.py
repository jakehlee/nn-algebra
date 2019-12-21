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

    model = models.mobilenet_v2(pretrained=True)
    model.cuda()
    model.eval()

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


    with open("mobile/fc.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc_out)


