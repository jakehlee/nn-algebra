#!/bin/bash
#fc
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2/fc.csv mobilev2 plane fc tl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2/fc.csv mobilev2 plane fc tr 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2/fc.csv mobilev2 plane fc c 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2/fc.csv mobilev2 plane fc br 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2/fc.csv mobilev2 plane fc bl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2-AA/fc.csv mobilev2AA plane fc tl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2-AA/fc.csv mobilev2AA plane fc tr 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2-AA/fc.csv mobilev2AA plane fc c 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2-AA/fc.csv mobilev2AA plane fc br 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2-AA/fc.csv mobilev2AA plane fc bl 0.965 1.0
#class
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2/class.csv mobilev2 plane class br 0.99 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/plane-mobilev2-AA/class.csv mobilev2AA plane class br 0.99 1.0
mv *d3.csv mobilev2/
mv *plot.png mobilev2/
