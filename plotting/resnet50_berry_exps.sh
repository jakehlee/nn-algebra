#!/bin/bash
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50/fc.csv resnet50 berry fc tl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50-AA/fc.csv resnet50AA berry fc tl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50/fc.csv resnet50 berry fc c 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50-AA/fc.csv resnet50AA berry fc c 0.965 1.0
mv *plot.png resnet50/
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50/fc.csv resnet50 berry fc br 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50-AA/fc.csv resnet50AA berry fc br 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50/class.csv resnet50 berry class br 0.99 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-resnet50-AA/class.csv resnet50AA berry class br 0.99 1.0
mv *plot.png resnet50/
