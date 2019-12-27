#!/bin/bash
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50/fc.csv resnet50 plane fc tl 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50-AA/fc.csv resnet50AA plane fc tl 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50/fc.csv resnet50 plane fc tr 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50-AA/fc.csv resnet50AA plane fc tr 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50/fc.csv resnet50 plane fc c 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50-AA/fc.csv resnet50AA plane fc c 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50/fc.csv resnet50 plane fc br 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50-AA/fc.csv resnet50AA plane fc br 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50/fc.csv resnet50 plane fc bl 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50-AA/fc.csv resnet50AA plane fc bl 0.91 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50/class.csv resnet50 plane class br 0.15 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-resnet50-AA/class.csv resnet50AA plane class br 0.15 1.0
mv *d3.csv resnet50/
mv *plot.png resnet50/
