import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import L2UniFrac as L2U
from helper import get_metadata_dict, get_profile_path_list, get_profile_name_list,\
	get_pheno_sample_dict, get_rep_sample_dict_wgs_component_wise_mean, get_taxonomy_in_order


def check_spread_by_boxplot(tax_list, rank):
    pdir = 'data/adenoma_266076/profiles'
    profile_path_list = get_profile_path_list(pdir)
    profile_name_list = get_profile_name_list(pdir)
    meta_file = 'data/hmgdb_adenoma_bioproject266076.csv'
    meta_dict = get_metadata_dict(meta_file)
    df = pd.DataFrame(columns=['environment', str(rank), 'abundance'])
    meta_col = []
    tax_col = []
    abund_col = []
    for i, file in enumerate(profile_path_list):
        print(file)
        if file.endswith('.profile'):
            with open(file, 'r') as f:
                for line in f:
                    if len(line.strip()) == 0 or line.startswith("#") or line.startswith("@"):
                        continue
                    line = line.rstrip('\n')
                    row_data = line.split('\t')
                    tax_rank = row_data[1]
                    taxpath = row_data[3]
                    tax_name = taxpath.split('|')[-1]
                    abund = row_data[-1]
                    #print(tax_rank, tax_name, abund)
                    if tax_rank == rank and tax_name in tax_list:
                        meta_col.append(meta_dict[profile_name_list[i]])
                        tax_col.append(tax_name)
                        abund_col.append(abund)
    df['environment'] = meta_col
    df[str(rank)] = tax_col
    df['abundance'] = abund_col
    print(df)
    df.to_csv('data/adenoma_266076/check_diffabund.txt', sep='\t')
    return df

def plot_df(df_file, rank):
    df = pd.read_table(df_file)
    sns.boxplot(x=rank, y='abundance', hue='environment', data=df, palette='Set2')
    plt.xticks(rotation=45)
    plt.savefig('data/adenoma_266076/check_diffabund_boxplot.png')
    plt.show()



df = check_spread_by_boxplot(['Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes'], 'phylum')
plot_df('data/adenoma_266076/check_diffabund.txt', 'phylum')