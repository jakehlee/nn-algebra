#!/bin/bash
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-bm100/fc.csv resnet18 plane fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-bm100/fc.csv resnet18 plane fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-bm100/fc.csv resnet18 plane fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-bm100/fc.csv resnet18 plane fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-bm100/fc.csv resnet18 plane fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-bm100/class.csv resnet18 plane class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-so100-2/fc.csv resnet18SO plane fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-so100-2/fc.csv resnet18SO plane fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-so100-2/fc.csv resnet18SO plane fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-so100-2/fc.csv resnet18SO plane fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-so100-2/fc.csv resnet18SO plane fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-so100-2/class.csv resnet18SO plane class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aa100/fc.csv resnet18AA plane fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aa100/fc.csv resnet18AA plane fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aa100/fc.csv resnet18AA plane fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aa100/fc.csv resnet18AA plane fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aa100/fc.csv resnet18AA plane fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aa100/class.csv resnet18AA plane class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aaso100/fc.csv resnet18AASO plane fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aaso100/fc.csv resnet18AASO plane fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aaso100/fc.csv resnet18AASO plane fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aaso100/fc.csv resnet18AASO plane fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aaso100/fc.csv resnet18AASO plane fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/plane-cifar-resnet18-aaso100/class.csv resnet18AASO plane class br 0.15 1.0

mv *d3.csv resnet18-100/
mv *plot.png resnet18-100/
