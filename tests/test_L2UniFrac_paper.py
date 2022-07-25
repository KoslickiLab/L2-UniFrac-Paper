import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('src')
sys.path.append('scripts')
import L2UniFrac as L2U
import partition_predict_16s as pp
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
    print(train_dict['skin'])

def test_decipher_label():
    prediction = [0,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1]
    group_name = 1
    training = ['a','b','c','d','e','f','g','h','i','j']
    meta_dict = {'a':{'body_site':'gut'},
                 'b':{'body_site':'gut'},
                 'c':{'body_site': 'gut'},
                 'd': {'body_site': 'gut'},
                 'e': {'body_site': 'skin'},
                 'f': {'body_site': 'skin'},
                 'g': {'body_site': 'gut'},
                 'h': {'body_site': 'skin'},
                 'i': {'body_site': 'skin'},
                 'j': {'body_site': 'skin'},
                 'k': {'body_site': 'gut'},
                 'l': {'body_site': 'skin'},
                 }
    sample_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11}
    label = pp.decipher_label_by_vote(prediction, training, group_name, meta_dict, sample_dict)
    print(label)

def test_get_sample_id_from_dict():
    t_dict = {'skin': {'sample1':[1,2,3], 'sample2':[3,4,5]}, 'gut':{'sample4':[4,5,6], 'sample5':[6,8,9]}}
    sample_lst = pp.get_sample_id_from_dict(t_dict)
    print(sample_lst)

def test_get_L2UniFrac_results():
    biom_file = 'data/biom/47422_otu_table.biom'
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    Tint, lint, nodes_in_order = parse_tree_file(tree_file)
    metadata_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
    train_dict, test_dict = pp.partition_samples(80, biom_file, tree_file, metadata_file, "body_site")
    meta_dict = extract_metadata(metadata_file)
    results = pp.get_L2UniFrac_accuracy_results(train_dict, test_dict, Tint, lint, nodes_in_order, meta_dict)
    print(results)


if __name__ == '__main__':
	#test_partition_sample()
    #test_decipher_label()
    test_get_sample_id_from_dict()