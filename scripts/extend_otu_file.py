import argparse
import os
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
except:
    pass
from src.parse_data import parse_otu_table, parse_tree_file
import pandas as pd

def argument_parser():
    parser = argparse.ArgumentParser(description="This function takes in a .tsv otu file and a metadata file,"
                                                 "produces representative samples for each phenotype under a specified"
                                                 "column in the metadata file, and writes the representative samples into"
                                                 "a new .tsv otu file.")
    parser.add_argument('-i', '--otu_file', type=str, required=True, help='Path to the input otu file in tsv format.')
    parser.add_argument('-t', '--tree_file', type=str, required=True, help='Path to tree file.')
    parser.add_argument('-o', '--output_file', type=str, help='File path to save the new otu file as.')
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    Tint, lint, nodes_in_order = parse_tree_file(args.tree_file)
    extended_df = parse_otu_table(args.otu_file, nodes_in_order, normalize=True)
    extended_df.to_csv(args.output_file, sep='\t')


if __name__ == "__main__":
    main()
