import argparse
import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import L1UniFrac as L1U
import L2UniFrac as L2U
from extract_data import parse_tree_file, extract_samples_direct
from helper import get_metadata_dict, get_meta_samples_dict, get_scatter_plot_for_L1_L2_vectors

def main():
    parser = argparse.ArgumentParser(description="Compare the correlation between an L1 average vector and a L2 average "
                                                 "vector (component wise)")
    parser.add_argument('-f', '--otu_file', type=str, default='data/1928_body_sites/47422_otu_table.biom')
    parser.add_argument('-m', '--meta_file', type=str, default='data/1928_body_sites/P_1928_65684500_raw_meta.txt')
    parser.add_argument('-o', '--output_prefix', type=str, required=True)

    args = parser.parse_args()
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    Tint, lint, nodes_in_order = parse_tree_file(tree_file)
    meta_dict = get_metadata_dict(args.meta_file, key_col="sample_name", val_col="body_site")
    meta_samples_dict = get_meta_samples_dict(meta_dict)
    sample_vector_dict, sample_ids = extract_samples_direct(args.otu_file, tree_file)
    #L1 average vectors
    L1_rep_sample_dict = L1U.get_L1_representative_sample_16s(sample_vector_dict, meta_samples_dict, Tint, lint,
                                                              nodes_in_order)
    #L2 average vectors
    L2_rep_sample_dict = L2U.get_representative_sample_16s(sample_vector_dict, meta_samples_dict, Tint, lint,
                                                           nodes_in_order)
    print(L1_rep_sample_dict)
    print(L2_rep_sample_dict)
    get_scatter_plot_for_L1_L2_vectors(L1_rep_sample_dict, L2_rep_sample_dict, args.output_prefix)


if __name__ == "__main__":
    main()