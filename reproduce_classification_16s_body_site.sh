#!/bin/bash
nohup python scripts/partition_predict_16s.py -m data/1928_body_sites/P_1928_65684500_raw_meta.txt -p env -bf data/1928_body_sites/47422_otu_table.biom -dm data/L2-UniFrac-Out.csv -n 10 -s data/1928_body_sites/16s_bodysites_classification_df.txt -c 4 &
#separate classification df into individual score types
#accuracy
cd data/1928_body_sites
head -1 16s_bodysites_classification_df.txt > classification_accuracy_df.txt
grep 'accuracy_score' 16s_bodysites_classification_df.txt >> classification_accuracy_df.txt 
#precision_micro
head -1 16s_bodysites_classification_df.txt > classification_precision_micro_df.txt
grep 'precision_micro' 16s_bodysites_classification_df.txt >> classification_precision_micro_df.txt 
#recall micro
head -1 16s_bodysites_classification_df.txt > classification_recall_micro_df.txt
grep 'recall_micro' 16s_bodysites_classification_df.txt >> classification_recall_micro_df.txt 
cd ..
cd ..
#get boxplot
python scripts/plot_df.py -f data/1928_body_sites/classification_accuracy_df.txt -t box -hue Method -x Score_type -y Score -s data/1928_body_sites/16s_classification_accuracy_scores.png
python scripts/plot_df.py -f data/1928_body_sites/classification_precision_micro_df.txt -t box -hue Method -x Score_type -y Score -s data/1928_body_sites/16s_classification_precision_micro_scores.png
python scripts/plot_df.py -f data/1928_body_sites/classification_recall_micro_df.txt -t box -hue Method -x Score_type -y Score -s data/1928_body_sites/16s_classification_recall_micro_scores.png
