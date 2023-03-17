import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')

import argparse
import numpy as np
from extract_data import parse_tree_file, extract_samples_direct
import L1UniFrac as L1U
from helper import get_metadata_dict, get_meta_samples_dict
import pandas as pd

def count_L1_UniFrac_negatives(meta_sample_dict, Tint, lint, nodes_in_order, outfile):
    df = pd.DataFrame(columns=['environment', '# of negative in average L1 vector'])
    neg_count_col = []
    env_col = []
    for pheno in meta_sample_dict:
        pushed_up_vectors = [L1U.push_up(x, Tint, lint, nodes_in_order) for x in meta_sample_dict[pheno]]
        average_vector = L1U.median_of_vectors(pushed_up_vectors)
        neg_count = len(list(filter(lambda x: (x < 0), average_vector)))
        env_col.append(pheno)
        neg_count_col.append(neg_count)
    df['environment'] = env_col
    df['# of negative in average vector'] = neg_count_col
    print(df)
    df.to_csv(outfile, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given an OTU file, get average L1-UniFrac vector and count how many '
                                                 'negative values in the vector.')
    parser.add_argument('-f', '--biom_file', type=str, default='data/1928_body_sites/47422_otu_table.biom')
    parser.add_argument('-m', '--meta_file', type=str, default='sample_vector_dict, sample_ids = extract_samples_direct(args.biom_file, tree_file')
    parser.add_argument('-s', '--save', type=str, default='data/1928_body_sites/L1_negative_counts')
    args = parser.parse_args()


    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    Tint, lint, nodes_in_order = parse_tree_file(tree_file)
    sample_vector_dict, sample_ids = extract_samples_direct(args.biom_file, tree_file)
    meta_dict = get_metadata_dict(args.meta_file, val_col=args.phenotype, key_col="sample_name")
    meta_sample_dict = get_meta_samples_dict(meta_dict)
    count_L1_UniFrac_negatives(meta_sample_dict, Tint, lint, nodes_in_order, args.save)
