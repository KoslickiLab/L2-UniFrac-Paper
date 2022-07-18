import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('src')
sys.path.append('scripts')
import L2UniFrac as L2U
import partition_predict as pp
from extract_data import extract_biom, extract_samples, extract_metadata, parse_tree_file, parse_envs
import os


def test_push_up_from_wgs_profile():
    print(L2U.push_up_from_wgs_profile('wgs-env0-sample24-reads.profile'))

def test_merge_efficiency():
    return

def test_merge_profiles_by_dir():
    L2U.merge_profiles_by_dir('profile_test')

def test_partition_sample():
    train_percentage=80
    biom_file = 'data/biom/47422_otu_table.biom'
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    metadata_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
    metadata_key = 'body_site'
    train_dict, test_dict = pp.partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)
    print(train_dict.keys())


if __name__ == '__main__':
	test_partition_sample()
