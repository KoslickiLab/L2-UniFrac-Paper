import argparse
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
import argparse
import L2UniFrac as L2U
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
except:
    pass
from src.parse_data import parse_otu_table, parse_tree_file
import itertools as it


def argument_parser():
    parser = argparse.ArgumentParser(description="Get pairwise L2-UniFrac matrix from otu table")
    parser.add_argument('-i', '--otu_file', type=str, required=True, help='Path to the input otu file in tsv format.')
    parser.add_argument('-t', '--tree_file', type=str, required=True, help='Path to tree file.')
    parser.add_argument('-o', '--output_file', type=str, help='File path to save the distance matrix file as.')
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    sample_vector_dict, sample_ids, otus = parse_otu_table(args.otu_file, normalize=True)
    Tint, lint, nodes_in_order = parse_tree_file(args.tree_file)
    dim = len(sample_ids)
    dist_matrix = np.zeros(shape=(dim, dim))
    for pair in it.combinations(sample_ids, 2): #all pairwise combinations
        sample_p, sample_q = sample_vector_dict[pair[0]], sample_vector_dict[pair[1]]
        unifrac = L2U.L2UniFrac_weighted_plain(Tint, lint, nodes_in_order, sample_p, sample_q)
        i = sample_ids.index(pair[0])
        j = sample_ids.index(pair[1])
        dist_matrix[i][j] = dist_matrix[j][i] = unifrac
    pd.DataFrame(data=dist_matrix, index=sample_ids, columns=sample_ids).to_csv(args.output_file, sep="\t")


if __name__ == '__main__':
    main()