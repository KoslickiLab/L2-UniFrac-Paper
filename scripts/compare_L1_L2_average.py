import argparse
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import L1UniFrac as L1U
import L2UniFrac as L2U
from extract_data import parse_tree_file, extract_samples_direct
from helper import get_metadata_dict, get_meta_samples_dict, get_scatter_plot_for_L1_L2_vectors
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compare the correlation between an L1 average vector and a L2 average "
                                                 "vector (component wise)")
    parser.add_argument('-f1', '--otu_file1', type=str, default='data/714_mouse/representative_otu_by_bodysite_L1.txt')
    parser.add_argument('-f2', '--otu_file2', type=str, default='data/714_mouse/representative_otu_by_bodysite.txt')
    parser.add_argument('-o', '--output_prefix', type=str, required=True, help="prefix including file path")

    args = parser.parse_args()
    L1_rep_sample_df = pd.read_table(args.otu_file1)
    L2_rep_sample_df = pd.read_table(args.otu_file2)
    L1_rep_sample_dict = dict()
    L2_rep_sample_dict = dict()
    for env in L2_rep_sample_df.columns:
        L1_rep_sample_dict[env] = L1_rep_sample_df[env]
        L2_rep_sample_dict[env] = L2_rep_sample_df[env]
    get_scatter_plot_for_L1_L2_vectors(L1_rep_sample_dict, L2_rep_sample_dict, args.output_prefix)


if __name__ == "__main__":
    main()