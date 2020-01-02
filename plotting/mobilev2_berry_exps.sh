#!/bin/bash
#fc
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2/fc.csv mobilev2 berry fc tl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2/fc.csv mobilev2 berry fc tr 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2/fc.csv mobilev2 berry fc c 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2/fc.csv mobilev2 berry fc br 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2/fc.csv mobilev2 berry fc bl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2-AA/fc.csv mobilev2AA berry fc tl 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2-AA/fc.csv mobilev2AA berry fc tr 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2-AA/fc.csv mobilev2AA berry fc c 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2-AA/fc.csv mobilev2AA berry fc br 0.965 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2-AA/fc.csv mobilev2AA berry fc bl 0.965 1.0
#class
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2/class.csv mobilev2 berry class br 0.99 1.0
python plot_cosine.py /home/jhl2195/nn-algebra/extraction/output/berry-mobilev2-AA/class.csv mobilev2AA berry class br 0.99 1.0
mv *d3.csv mobilev2/
mv *plot.png mobilev2/
