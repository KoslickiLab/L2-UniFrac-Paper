import pandas as pd
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
from extract_data import parse_tree_file, extract_samples_direct
from src.helper import get_meta_samples_dict, get_metadata_dict

def argument_parser():
    parser = argparse.ArgumentParser(description="This function takes in a .tsv otu file and a metadata file,"
                                                 "produces representative samples for each phenotype under a specified"
                                                 "column in the metadata file, and writes the representative samples into"
                                                 "a new .tsv otu file.")
    parser.add_argument('-i', '--otu_file', type=str, required=True, help='Path to the input otu file in tsv format. The'
                                                                          'number of OTUs in the table must be the same'
                                                                          'as that of the tree file. If not, first run'
                                                                          'extend_otu_file.py')
    parser.add_argument('-t', '--tree_file', type=str, help='Path to tree file.', default='data/trees/'
                                                                                                         'gg_13_5_otus_99_annotated.tree')
    parser.add_argument('-o', '--output_file', type=str, help='File path to save the new otu file as.')
    parser.add_argument('-m', '--meta_file', type=str, required=True, help='Path to metadata file.')
    parser.add_argument('-k', '--key', type=str, help="Key column name in metadata file. Usually sample ids.", default=
                        'sample_name')
    parser.add_argument('-v', '--val', type=str, help="Value column name in metadata file. e.g. diagnosis, "
                                                      "environment, etc.", default='body_site')
    parser.add_argument('-L1', '--L1_average', type=bool, help="If given, L1 average will also be produced.")
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    Tint, lint, nodes_in_order = parse_tree_file(args.tree_file)
    if args.otu_file.endswith('.tsv'):
        df = pd.read_table(args.otu_file, sep='\t', index_col=0)
        sample_vector_dict = df.to_dict(orient='list')
    elif args.otu_file.endswith('.biom'):
        sample_vector_dict, sample_ids = extract_samples_direct(args.otu_file, args.tree_file)
    #push up all the samples
    simple_meta_dict = get_metadata_dict(args.meta_file, val_col=args.val, key_col=args.key)
    meta_samples_dict = get_meta_samples_dict(simple_meta_dict)
    rep_sample_dict = L2U.get_representative_sample_16s(sample_vector_dict, meta_samples_dict, Tint, lint, nodes_in_order)
    df = pd.DataFrame.from_dict(rep_sample_dict)
    df.index = nodes_in_order
    df.to_csv(args.output_file, sep='\t')
    if args.L1_average:
        L1_rep_sample_dict = L1U.get_L1_representative_sample_16s(sample_vector_dict, meta_samples_dict, Tint, lint,
                                                               nodes_in_order)
        df = pd.DataFrame.from_dict(L1_rep_sample_dict)
        df.index = nodes_in_order
        new_outfile_name = args.output_file.split('.')[0] + "_L1.txt"
        df.to_csv(new_outfile_name, sep='\t')
    return


if __name__ == '__main__':
    main()