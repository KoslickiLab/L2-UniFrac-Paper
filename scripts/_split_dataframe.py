import argparse
import pandas as pd
import sys
import os
sys.path.append('src')
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
from helper import get_metadata_dict, get_meta_samples_dict
from extract_data import extract_samples_direct, parse_tree_file
import L2UniFrac as L2U

def parse_arguments():
    parser = argparse.ArgumentParser(description="Given an otu file and metadata, this script does the following:"
                                                 "1. group samples by a specific environment type.e.g. body site."
                                                 "2. compute the representative sample for each environment."
                                                 "3. for each environment, write all the samples and the representative"
                                                 "sample into the same .tsv file and output in the output directory."
                                                 "4. for each environment, create a metadata file and output to the same"
                                                 "directory.")
    parser.add_argument('-f', '--biom_file', type=str, help="OTU file to be split.")
    parser.add_argument('-o', '--out_dir', type=str, help="Output directory to write to.")
    parser.add_argument('-e', '--env_name', type=str, help="Environment name.e.g. body site, treatment")
    parser.add_argument('-m', '--metadata_file', type=str, help="Metadata file.")
    parser.add_argument('-t', '--tree_file', type=str, help="Tree file", nargs='?', default='data/trees/gg_13_5_otus_99_annotated.tree')
    return parser

def split_df(sample_vector_dict, meta_samples_dict, rep_sample_dict, nodes_in_order, out_dir):
    for phenotype in meta_samples_dict:
        print(phenotype)
        samples_in_this_pheno = dict()
        this_meta_dict = dict()
        out_otu_file = os.path.join(out_dir, phenotype + '_and_representative.tsv')
        for sample in meta_samples_dict[phenotype]:
            if sample in sample_vector_dict:
                samples_in_this_pheno[sample] = sample_vector_dict[sample]
                this_meta_dict[sample] = phenotype
        samples_in_this_pheno[phenotype] = rep_sample_dict[phenotype]
        this_meta_dict[phenotype] = "representative_sample"
        df = pd.Dataframe.from_dict(samples_in_this_pheno)
        df.set_index(nodes_in_order, inplace=True)
        meta_df = pd.DataFrame.from_dict(meta_samples_dict)
        out_meta_file = os.path.join(out_dir, phenotype + '_and_representative_meta.tsv')
        meta_df.to_csv(out_meta_file, sep='\t')
        print(df.head())
        df.to_csv(out_otu_file, sep='\t', index=0)


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    meta_dict = get_metadata_dict(args.metadata_file, val_col=args.env_name, key_col='sample_name')
    Tint, lint, nodes_in_order = parse_tree_file(args.tree_file)
    meta_samples_dict = get_meta_samples_dict(meta_dict)
    sample_vector_dict, sample_ids = extract_samples_direct(args.biom_file, args.tree_file)
    rep_sample_dict = L2U.get_representative_sample_16s(sample_vector_dict, meta_samples_dict, Tint, lint,
                                                        nodes_in_order)
    split_df(sample_vector_dict, meta_samples_dict, rep_sample_dict, nodes_in_order, args.out_dir)


if __name__ == "__main__":
    main()