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
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(root=img_dir, transform=img_transform)
    dataset_loader = data.DataLoader(dataset, batch_size=BATCH)

    model = models.resnet50(pretrained=True)
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


