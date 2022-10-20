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
from src.helper import get_meta_samples_dict, get_metadata_dict

def argument_parser():
    parser = argparse.ArgumentParser(description="This function takes in a .tsv otu file and a metadata file,"
                                                 "produces representative samples for each phenotype under a specified"
                                                 "column in the metadata file, and writes the representative samples into"
                                                 "a new .tsv otu file.")
    parser.add_argument('-i', '--otu_file', type=str, required=True, help='Path to the input otu file in tsv format.')
    parser.add_argument('-t', '--tree_file', type=str, required=True, help='Path to tree file.')
    parser.add_argument('-o', '--output_file', type=str, help='File path to save the new otu file as.')
    parser.add_argument('-m', '--meta_file', type=str, required=True, help='Path to metadata file.')
    parser.add_argument('-k', '--key', type=str, help="Key column name in metadata file. Usually sample ids.")
    parser.add_argument('-v', '--val', type=str, help="Value column name in metadata file. e.g. diagnosis, environment, etc.")
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    Tint, lint, nodes_in_order = parse_tree_file(args.tree_file)
    sample_vector_dict, sample_ids = parse_otu_table(args.otu_file, nodes_in_order, normalize=True)
    #push up all the samples
    simple_meta_dict = get_metadata_dict(args.meta_file, val_col=args.val, key_col=args.key)
    meta_samples_dict = get_meta_samples_dict(simple_meta_dict)
    rep_sample_dict = L2U.get_rep_sample_dict(sample_vector_dict, meta_samples_dict, Tint, lint, nodes_in_order)
    df = pd.DataFrame(rep_sample_dict, columns=sample_ids, index=nodes_in_order)
    for sample in sample_ids:
        df[sample] = sample_vector_dict[sample]
    df.to_csv(args.output_file, sep='\t')
    return


if __name__ == '__main__':
    main()