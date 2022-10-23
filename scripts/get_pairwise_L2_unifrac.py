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
import itertools as it
from src.parse_data import parse_tree_file
import pandas as pd

def argument_parser():
    parser = argparse.ArgumentParser(description="Get pairwise L2-UniFrac matrix from otu table")
    parser.add_argument('-i', '--otu_file', type=str, required=True, help='Path to the input otu file in tsv format. The'
                                                                          'number of OTUs in the table must be the same'
                                                                          'as that of the tree file. If not, first run'
                                                                          'extend_otu_file.py')
    parser.add_argument('-t', '--tree_file', type=str, required=True, help='Path to tree file.')
    parser.add_argument('-o', '--output_file', type=str, help='File path to save the distance matrix file as.')
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    Tint, lint, nodes_in_order = parse_tree_file(args.tree_file)
    df = pd.read_table(args.otu_file, sep='\t')
    sample_ids = df.columns.tolist()
    sample_vector_dict = df.to_dict(orient='list')
    dim = len(sample_ids)
    dist_matrix = np.zeros(shape=(dim, dim))
    for pair in it.combinations(sample_ids, 2): #all pairwise combinations
        sample_p, sample_q = sample_vector_dict[pair[0]], sample_vector_dict[pair[1]]
        i = sample_ids.index(pair[0])
        j = sample_ids.index(pair[1])
        print(sample_p)
        print(sample_q)
        print(np.dtype(sample_p))
        print(np.dtype(sample_q))
        unifrac = L2U.L2UniFrac_weighted_plain(Tint, lint, nodes_in_order, sample_p, sample_q)
        dist_matrix[i][j] = dist_matrix[j][i] = unifrac
    pd.DataFrame(data=dist_matrix, index=sample_ids, columns=sample_ids).to_csv(args.output_file, sep="\t")


if __name__ == '__main__':
    main()