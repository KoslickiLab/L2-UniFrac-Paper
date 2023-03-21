import numpy as np
import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import argparse
import L2UniFrac as L2U
import L1UniFrac as L1U
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
except:
    pass
import itertools as it
from extract_data import parse_tree_file
import pandas as pd

def argument_parser():
    parser = argparse.ArgumentParser(description="Get pairwise L1 or L2-UniFrac matrix from otu table")
    parser.add_argument('-i', '--otu_file', type=str, required=True, help='Path to the input otu file in tsv format. The'
                                                                          'number of OTUs in the table must be the same'
                                                                          'as that of the tree file. If not, first run'
                                                                          'extend_otu_file.py')
    parser.add_argument('-t', '--tree_file', type=str, help='Path to tree file.', default='data/trees/gg_13_5_otus_99_annotated.tree')
    parser.add_argument('-o', '--output_file', type=str, help='File path to save the distance matrix file as.')
    parser.add_argument('-L1', '--L1UniFrac', type=int, default=0, help='pairwise L1 UniFrac instead of L2. Default L2. L1 if set to 1.')
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    Tint, lint, nodes_in_order = parse_tree_file(args.tree_file)
    df = pd.read_table(args.otu_file, sep='\t', index_col=0) #use otus as index column
    print(df.head())
    sample_ids = df.columns.tolist()
    print(sample_ids)
    sample_vector_dict = df.to_dict(orient='list')
    dim = len(sample_ids)
    dist_matrix = np.zeros(shape=(dim, dim))
    if args.L1 == 1:
        df = L1U.pairwise_L1EMDUniFrac_weighted(sample_vector_dict, Tint, lint, nodes_in_order)
        df.to_csv(args.output_file, sep='\t')
    else:
        for pair in it.combinations(sample_ids, 2): #all pairwise combinations
            sample_p, sample_q = sample_vector_dict[pair[0]], sample_vector_dict[pair[1]]
            i = sample_ids.index(pair[0])
            j = sample_ids.index(pair[1])
            unifrac = L2U.L2UniFrac_weighted_plain(Tint, lint, nodes_in_order, sample_p, sample_q)
            dist_matrix[i][j] = dist_matrix[j][i] = unifrac
        pd.DataFrame(data=dist_matrix, index=sample_ids, columns=sample_ids).to_csv(args.output_file, sep="\t")


if __name__ == '__main__':
    main()