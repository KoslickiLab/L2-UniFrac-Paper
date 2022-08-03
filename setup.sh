#!/bin/sh
git clone https://github.com/KoslickiLab/L2-UniFrac.git
conda create --name L2UniFrac
conda activate L2UniFrac
pip install motu-profiler
motus downloadDB
pip install biom-format
pip install -U dendropy
conda -y install numpy
pip install matplotlib
pip install scikit-learn-extra
pip install seaborn

