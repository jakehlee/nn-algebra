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

from models_exp.so_resnet18 import resnet18Shortout

BATCH = 64

def usage():
    print("Usage: python ex-model.py img_dir out_dir class_id")
    sys.exit(0)

if __name__ == "__main__":
    img_dir = sys.argv[1]
    out_dir = sys.argv[2]
    class_id = int(sys.argv[3])

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    img_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])])
    
    dataset = datasets.ImageFolder(root=img_dir, transform=img_transform)
    dataset_loader = data.DataLoader(dataset, batch_size=BATCH)

    model = resnet18Shortout()
    model.load_state_dict(torch.load('/home/jhl2195/nn-algebra/shortout/chkpt/so_weights_20200102_231018_39.pth'))

    model.cuda()
    model.eval()

    fc_out = []
    class_out = []
    for i, (img_batch, _) in enumerate(dataset_loader,0):
        print("Extracting batch {}".format(i))
        img_batch = img_batch.cuda()
        ptr = i * BATCH

        fc_buf = model(img_batch)
        class_buf = F.softmax(fc_buf, dim=1).tolist()
        fc_buf = fc_buf.tolist()

        for j in range(len(fc_buf)):
            img_name = os.path.split(dataset.imgs[ptr+j][0])[-1]
            fc_out.append([img_name] + fc_buf[j])
            class_out.append([img_name, class_buf[j][class_id]])

    with open(os.path.join(out_dir, "fc.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(fc_out)

    with open(os.path.join(out_dir, "class.csv"), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(class_out)


