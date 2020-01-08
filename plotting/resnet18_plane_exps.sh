#!/bin/bash
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18/fc.csv resnet18 plane fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO/fc.csv resnet18SO plane fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO2/fc.csv resnet18SO2 plane fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18/fc.csv resnet18 plane fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO/fc.csv resnet18SO plane fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO2/fc.csv resnet18SO2 plane fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18/fc.csv resnet18 plane fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO/fc.csv resnet18SO plane fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO2/fc.csv resnet18SO2 plane fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18/fc.csv resnet18 plane fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO/fc.csv resnet18SO plane fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO2/fc.csv resnet18SO2 plane fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18/fc.csv resnet18 plane fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO/fc.csv resnet18SO plane fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO2/fc.csv resnet18SO2 plane fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18/class.csv resnet18 plane class br 0.15 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO/class.csv resnet18SO plane class br 0.15 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/plane-cifar-resnet18-SO2/class.csv resnet18SO2 plane class br 0.15 1.0
mv *d3.csv resnet18/
mv *plot.png resnet18/
