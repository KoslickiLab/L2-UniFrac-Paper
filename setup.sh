#!/bin/sh
git clone https://github.com/KoslickiLab/L2-UniFrac.git
conda create --name L2UniFrac
conda activate L2UniFrac
pip install motu-profiler
motus downloadDB
pip install biom-format
pip install -U dendropy
