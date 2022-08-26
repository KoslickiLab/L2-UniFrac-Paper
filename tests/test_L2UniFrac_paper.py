import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('src')
sys.path.append('scripts')
import L2UniFrac as L2U
import partition_predict_16s as pp
from extract_data import extract_biom, extract_samples, extract_sample_metadata, parse_tree_file, parse_envs, extract_biom_samples, extract_metadata
import partition_predict_wgs as pp2
from copy import deepcopy
import os
import clustering_16s as clustering
from sklearn_extra.cluster import KMedoids
import pandas as pd

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
    print("len  of Tint:", len(Tint))
    print("len of nodes in order:", len(nodes_in_order))
    metadata_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
    train_dict, test_dict = pp.partition_samples(80, biom_file, tree_file, metadata_file, "body_site")
    meta_dict = extract_metadata(metadata_file)
    results = pp.get_L2UniFrac_accuracy_results(train_dict, test_dict, Tint, lint, nodes_in_order)
    print(results)

def test_get_wgs_metadict():
    meta_file = 'data/hmgdb_adenoma_bioproject266076.csv'
    meta_dict = pp2.get_metadata_dict(meta_file)
    print(meta_dict)

def test_train_test_split():
    meta_dict = pp2.get_metadata_dict('data/hmgdb_adenoma_bioproject266076.csv')
    samples_train, samples_test, targets_train, targets_test = pp2.partition_sample(meta_dict)
    print(len(samples_train))
    print(len(samples_test))
    print(len(targets_train))
    print(len(targets_test))
    print(targets_test)
    print(samples_test)

def test_merge_profile():
    profile_path1 = 'tests/profile_test/wgs-env0-sample0-reads.profile'
    profile_path2 = 'tests/profile_test/wgs-env1-sample10-reads.profile'
    profile_list1 = L2U.open_profile_from_tsv(profile_path1, False)
    name, metadata, profile = profile_list1[0]
    #print(profile)
    profile1 = L2U.Profile(sample_metadata=metadata, profile=profile)
    profile_list2 = L2U.open_profile_from_tsv(profile_path2, False)
    name, metadata, profile = profile_list2[0]
    profile2 = L2U.Profile(sample_metadata=metadata, profile=profile)
    (Tint, lint, nodes_in_order, nodes_to_index, P) = profile1.make_unifrac_input_and_normalize()

    print(len(profile1._data))

    profile3 = deepcopy(profile1)
    profile3.merge(profile2)
    print(len(profile1._data))
    print(len(profile3._data))

def test_get_rep_sample_from_profiles():
    profile_path1 = 'tests/profile_test/wgs-env0-sample0-reads.profile'
    profile_path2 = 'tests/profile_test/wgs-env1-sample10-reads.profile'
    profile_path3 = 'tests/profile_test/wgs-env1-sample4-reads.profile'
    profile_path4 = 'tests/profile_test/wgs-env0-sample21-reads.profile'
    #profile_lst = [profile_path1, profile_path2, profile_path3, profile_path4]
    profile_lst = [profile_path1, profile_path2, profile_path3]
    rep_vector, Tint, lint, nodes_in_order = L2U.get_representative_sample_wgs(profile_lst)
    print(rep_vector)

def test_wgs_L2UniFrac():
    profile_path1 = 'tests/profile_test/test_profile1.profile'
    profile_path2 = 'tests/profile_test/test_profile2.profile'
    profile_list = [profile_path1, profile_path2]
    profile_list1 = L2U.open_profile_from_tsv(profile_path1, False)
    name1, metadata1, profile1 = profile_list1[0]
    real_profile1 = L2U.Profile(sample_metadata=metadata1, profile=profile1)
    (Tint1, lint1, nodes_in_order1, nodes_to_index1, P1) = real_profile1.make_unifrac_input_and_normalize()
    pushed_1 = L2U.push_up(P1, Tint1, lint1, nodes_in_order1)
    #print(pushed_1)
    profile_list2 = L2U.open_profile_from_tsv(profile_path2, False)
    name2, metadata2, profile2 = profile_list2[0]
    real_profile2 = L2U.Profile(sample_metadata=metadata2, profile=profile2)
    (Tint2, lint2, nodes_in_order2, nodes_to_index2, P2) = real_profile2.make_unifrac_input_and_normalize()
    pushed_2 = L2U.push_up(P2, Tint2, lint2, nodes_in_order2)
    #print(pushed_2)
    real_profile1.merge(real_profile2)
    (Tint, lint, nodes_in_order, nodes_to_index, P) = real_profile1.make_unifrac_input_and_normalize()
    print(nodes_to_index)
    #print(Tint)
    #print(lint)
    #print(nodes_in_order)
    #print(nodes_to_index)
    #print(P)
    rep_vector, Tint, lint, nodes_in_order, nodes_to_index = L2U.get_representative_sample_wgs(profile_list)
    print(rep_vector)
    print(nodes_to_index)
    #print(all_taxids_list)
    #print(nodes_to_index)

def test_get_wgs_clustering_results():
    meta_dict = pp2.get_metadata_dict('data/hmgdb_adenoma_bioproject266076.csv')
    samples_train, samples_test, targets_train, targets_test = pp2.partition_sample(meta_dict, 0)
    print(targets_train)
    kmedoids_prediction, sample_ids = pp2.get_KMedoids_prediction('data/adenoma_266076/adenoma_pairwise_L1UniFrac.txt',2)
    sample_index_dict = pp2.get_index_dict(sample_ids)
    print(kmedoids_prediction)
    results = pp2.get_clustering_scores(kmedoids_prediction, samples_test, meta_dict, sample_index_dict)
    print(results)

def test_get_KMedoids_prediction():
    dmatrix_file = 'data/adenoma_266076/adenoma_pairwise_L1UniFrac.txt'
    kmedoids_prediction, sample_ids = pp2.get_KMedoids_prediction(dmatrix_file,3)
    print(kmedoids_prediction[:10])
    print(sample_ids[:10])
    meta_dict = pp2.get_metadata_dict('data/hmgdb_adenoma_bioproject266076.csv')

    merged_prediction, updated_group_label_dict, sample_ids = pp2.try_cluster(2, 30, 3, 'kmedoids', dmatrix_file, meta_dict)
    print(meta_dict)
    print('merged prediction', merged_prediction)

def test_get_merged_clusters():
    dmatrix_file = 'data/adenoma_266076/adenoma_pairwise_L1UniFrac.txt'
    profile_dir = 'data/adenoma_266076/profiles'
    profile_path_lst = [os.path.join(profile_dir, file) for file in os.listdir(profile_dir)]
    Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_lst)

    meta_dict = pp2.get_metadata_dict('data/hmgdb_adenoma_bioproject266076.csv')
    #merged_prediction, updated_group_label_dict, sample_ids = pp2.try_cluster(2, 20, 3, "kmedoids", dmatrix_file, meta_dict)
    #print(updated_group_label_dict)
    all_samples = sample_id = list(meta_dict.keys())
    all_samples_paths = [profile_dir + '/' + sample + '.profile' for sample in all_samples]
    sample_vector_dict = L2U.merge_profiles_by_dir(all_samples_paths, nodes_to_index)
    merged_prediction, updated_group_label_dict, sample_ids = pp2.try_cluster(2, 20, 3, "kmeans", sample_vector_dict, meta_dict)
    print(updated_group_label_dict)

def test_16s_cluster():
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    Tint, lint, nodes_in_order = parse_tree_file(tree_file)
    meta_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
    biom_file = 'data/biom/47422_otu_table.biom'
    dmatrix_file = 'data/L1-UniFrac-Out.csv'
    sample_vector = extract_biom_samples(biom_file)
    #print(list(sample_vector.keys())[0])
    #print(list(sample_vector.values())[0])

    for k in sample_vector:
        L2U.push_up(sample_vector[k], Tint, lint, nodes_in_order)

def test_combine_train_test():
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    Tint, lint, nodes_in_order = parse_tree_file(tree_file)
    meta_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
    biom_file = 'data/biom/47422_otu_table.biom'
    dmatrix_file = 'data/L1-UniFrac-Out.csv'
    train_dict, test_dict = pp.partition_samples(80, biom_file, tree_file, meta_file, 'body_site')
    sample_dict = clustering.combine_train_test(train_dict, test_dict)
    print(len(sample_dict))
    print(len(list(sample_dict.values())[0]))

def test_get_KMedoids_clustering_score():
    dmatrix_file = 'data/L1-UniFrac-Out.csv'
    meta_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
    biom_file = 'data/biom/47422_otu_table.biom'
    meta_dict = extract_metadata(meta_file)
    sample_ids = extract_samples(biom_file)
    distance_matrix = pd.read_csv(dmatrix_file, header=None)
    kmedoids_prediction = KMedoids(n_clusters=5, metric='precomputed', method='pam',
                                   init='heuristic').fit_predict(distance_matrix)
    score = clustering.get_KMedoids_clustering_score(dmatrix_file, 5, sample_ids, meta_dict)
    print(score)


if __name__ == '__main__':
	#test_partition_sample()
    #test_decipher_label()
    #test_get_L2UniFrac_results()
    #test_get_wgs_metadict()
    #test_train_test_split()
    #test_merge_profile()
    #test_get_rep_sample_from_profiles()
    #test_wgs_L2UniFrac()
    #test_get_KMedoids_prediction()
    #test_get_wgs_clustering_results()
    #test_get_merged_clusters()
    #test_16s_cluster()
    #test_combine_train_test()
    test_get_KMedoids_clustering_score()