#!/bin/bash
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18/fc.csv resnet18 dog fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18/fc.csv resnet18 dog fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18/fc.csv resnet18 dog fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18/fc.csv resnet18 dog fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18/fc.csv resnet18 dog fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18/class.csv resnet18 dog class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-so/fc.csv resnet18SO dog fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-so/fc.csv resnet18SO dog fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-so/fc.csv resnet18SO dog fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-so/fc.csv resnet18SO dog fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-so/fc.csv resnet18SO dog fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-so/class.csv resnet18SO dog class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-aa/fc.csv resnet18AA dog fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-aa/fc.csv resnet18AA dog fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-aa/fc.csv resnet18AA dog fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-aa/fc.csv resnet18AA dog fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-aa/fc.csv resnet18AA dog fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-aa/class.csv resnet18AA dog class br 0.15 1.0

python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-soaa/fc.csv resnet18SOAA dog fc tl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-soaa/fc.csv resnet18SOAA dog fc tr 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-soaa/fc.csv resnet18SOAA dog fc c 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-soaa/fc.csv resnet18SOAA dog fc br 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-soaa/fc.csv resnet18SOAA dog fc bl 0.93 1.0
python plot_cosine_png.py /home/jhl2195/nn-algebra/extraction/output/cifar-exps/dog-cifar-resnet18-soaa/class.csv resnet18SOAA dog class br 0.15 1.0

mv *d3.csv resnet18-100/
mv *plot.png resnet18-100/
