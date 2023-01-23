#!/bin/bash

[ -d data/time_clustering ] || mkdir -p data/results/time_clustering

array=( 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 )

for i in "${array[@]}"
do
	python ./time_clustering.py -size $i -s data/results/time_clustering/"size$i.txt"
done

python scripts/_combine_df.py -d data/results/time_clustering -s data/results/time_clustering_combined_df.txt

python scripts/plot_df.py -f data/results/time_clustering/time_clustering_combined_df.txt -hue Method -x Sample_size -y Time -s data/results/time_clustering.png -t line
python scripts/plot_df.py -f data/results/time_clustering/time_clustering_combined_df.txt -hue Method -x Sample_size -y Fowlkes_Mallows_score -s data/results/score_clustering.png -t line -ylim 0. 1.
