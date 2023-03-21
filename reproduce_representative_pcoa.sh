#!/bin/bash

#get otu files for each phenotype + representative sample for treatment and for body site
python scripts/_split_dataframe.py -f data/714_mouse/extended_otu_table.biom -m data/714_mouse/714_20220510-094301.txt -e treatment -o data/714_mouse

#get pairwise distance matrix for each file
python scripts/get_pairwise_L2_unifrac.py -i data/714_mouse/untreated_soil_and_representative.tsv -t data/trees/gg_13_5_otus_99_annotated.tree -o data/714_mouse/pairwise_untreated_soil_and_representative.tsv
python scripts/get_pairwise_L2_unifrac.py -i data/714_mouse/control_and_representative.tsv -t data/trees/gg_13_5_otus_99_annotated.tree -o data/714_mouse/pairwise_control_and_representative.tsv
python scripts/get_pairwise_L2_unifrac.py -i data/714_mouse/pre_mortem_and_representative.tsv -t data/trees/gg_13_5_otus_99_annotated.tree -o data/714_mouse/pairwise_pre_mortem_and_representative.tsv
python scripts/get_pairwise_L2_unifrac.py -i data/714_mouse/fecal_and_representative.tsv -t data/trees/gg_13_5_otus_99_annotated.tree -o data/714_mouse/pairwise_fecal_and_representative.tsv
python scripts/get_pairwise_L2_unifrac.py -i data/714_mouse/sterile_soil_and_representative.tsv -t data/trees/gg_13_5_otus_99_annotated.tree -o data/714_mouse/pairwise_sterile_soil_and_representative.tsv

#get pcoa plot
#sterial_soil
python scripts/plot_df.py -t pcoa -f data/714_mouse/pairwise_sterile_soil_and_representative.tsv -m data/714_mouse/sterile_soil_and_representative_meta.tsv -env environment -s data/714_mouse/sterile_soil_and_representative_pcoa.png -cmap coolwarm
#pre_mortem
python scripts/plot_df.py -t pcoa -f data/714_mouse/pairwise_pre_mortem_and_representative.tsv -m data/714_mouse/pre_mortem_and_representative_meta.tsv -env environment -s data/714_mouse/pre_mortem_and_representative_pcoa.png -cmap coolwarm
#untreated_soil
python scripts/plot_df.py -t pcoa -f data/714_mouse/pairwise_untreated_soil_and_representative.tsv -m data/714_mouse/untreated_soil_and_representative_meta.tsv -env environment -s data/714_mouse/untreated_soil_and_representative_pcoa.png -cmap coolwarm
#control
python scripts/plot_df.py -t pcoa -f data/714_mouse/pairwise_control_and_representative.tsv -m data/714_mouse/control_and_representative_meta.tsv -env environment -s data/714_mouse/control_and_representative_pcoa.png -cmap coolwarm
#fecal
python scripts/plot_df.py -t pcoa -f data/714_mouse/pairwise_fecal_and_representative.tsv -m data/714_mouse/fecal_and_representative_meta.tsv -env environment -s data/714_mouse/fecal_and_representative_pcoa.png -cmap coolwarm

#2D pcoa
python scripts/plot_df.py -t 2Dpcoa -f data/714_mouse/pairwise_L2UniFrac_all.tsv -m data/714_mouse/metadata_representative_treatment.tsv -env treatment -cmap coolwarm -s data/714_mouse/treatment_all_pcoa_2D.png
python scripts/plot_df.py -t 2Dpcoa -f data/714_mouse/pairwise_control_and_representative.tsv -m data/714_mouse/control_and_representative_meta.tsv -env environment -s data/714_mouse/control_and_representative_pcoa_2D.png -cmap coolwarm
python scripts/plot_df.py -t 2Dpcoa -f data/714_mouse/pairwise_pre_mortem_and_representative.tsv -m data/714_mouse/pre_mortem_and_representative_meta.tsv -env environment -s data/714_mouse/pre_mortem_and_representative_pcoa_2D.png -cmap coolwarm
python scripts/plot_df.py -t 2Dpcoa -f data/714_mouse/pairwise_untreated_soil_and_representative.tsv -m data/714_mouse/untreated_soil_and_representative_meta.tsv -env environment -s data/714_mouse/untreated_soil_and_representative_pcoa_2D.png -cmap coolwarm
python scripts/plot_df.py -t 2Dpcoa -f data/714_mouse/pairwise_sterile_soil_and_representative.tsv -m data/714_mouse/sterile_soil_and_representative_meta.tsv -env environment -s data/714_mouse/sterile_soil_and_representative_pcoa_2D.png -cmap coolwarm
python scripts/plot_df.py -t 2Dpcoa -f data/714_mouse/pairwise_fecal_and_representative.tsv -m data/714_mouse/fecal_and_representative_meta.tsv -env environment -s data/714_mouse/fecal_and_representative_pcoa_2D.png -cmap coolwarm