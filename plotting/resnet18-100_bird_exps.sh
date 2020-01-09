#!/bin/bash
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18/fc.csv resnet18 bird fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18/fc.csv resnet18 bird fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18/fc.csv resnet18 bird fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18/fc.csv resnet18 bird fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18/fc.csv resnet18 bird fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18/class.csv resnet18 bird class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-so/fc.csv resnet18SO bird fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-so/fc.csv resnet18SO bird fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-so/fc.csv resnet18SO bird fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-so/fc.csv resnet18SO bird fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-so/fc.csv resnet18SO bird fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-so/class.csv resnet18SO bird class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-aa/fc.csv resnet18AA bird fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-aa/fc.csv resnet18AA bird fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-aa/fc.csv resnet18AA bird fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-aa/fc.csv resnet18AA bird fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-aa/fc.csv resnet18AA bird fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-aa/class.csv resnet18AA bird class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-soaa/fc.csv resnet18SOAA bird fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-soaa/fc.csv resnet18SOAA bird fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-soaa/fc.csv resnet18SOAA bird fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-soaa/fc.csv resnet18SOAA bird fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-soaa/fc.csv resnet18SOAA bird fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/bird-cifar-resnet18-soaa/class.csv resnet18SOAA bird class br 0.15 1.0

mv *d3.csv resnet18-100/
mv *plot.png resnet18-100/
