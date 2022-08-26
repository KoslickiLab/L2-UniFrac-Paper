import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
import argparse
import L2UniFrac as L2U
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from sklearn_extra.cluster import KMedoids
from extract_data import extract_biom, extract_samples, extract_metadata, parse_tree_file, parse_envs, extract_biom_samples
from math import floor

#Compare clustering among KMedoids, Kmeans and L2UniFrac
def extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key):
	nodes_samples = extract_biom(biom_file)
	_, _, nodes_in_order = parse_tree_file(tree_file)
	(nodes_weighted, samples_temp) = parse_envs(nodes_samples, nodes_in_order)
	sample_ids = extract_samples(biom_file)

	metadata = extract_metadata(metadata_file)
	group_name_samples = {}
	for sample in sample_ids:
		if metadata[sample][metadata_key] not in group_name_samples:
			group_name_samples[metadata[sample][metadata_key]] = {}
		group_name_samples[metadata[sample][metadata_key]][sample] = nodes_weighted[sample]

	return group_name_samples, sample_ids, list(group_name_samples.keys())

def get_KMedoids_clustering_score(dmatrix_file, n_clusters, sample_ids, meta_dict):
	distance_matrix = pd.read_csv(dmatrix_file, header=0, index_col=0, sep='\t')
	labels = get_true_label(meta_dict, sample_ids)
	print(labels)
	kmedoids_prediction = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='heuristic').fit_predict(distance_matrix)
	return fowlkes_mallows_score(kmedoids_prediction, labels)

def get_L2_clustering_score(L2_samples, n_clusters, meta_dict):
	all_vectors = list(L2_samples.values())
	sample_ids = list(L2_samples.keys())
	labels = get_true_label(meta_dict, sample_ids)
	kmeans_predict = KMeans(n_clusters=n_clusters).fit_predict(all_vectors)
	return fowlkes_mallows_score(kmeans_predict, labels)

def get_true_label(meta_dict, sample_ids):
	'''

	:param meta_dict: {sample_id: 'body_site': body_site}
	:param sample_ids: list of sample ids found in meta_dict
	:return: a list of true labels
	'''
	true_labels = [meta_dict[i] for i in sample_ids]
	return true_labels

def push_up_all(sample_vectors, Tint, lint, nodes_in_order):
	pushed_up_dict = dict()
	for sample in sample_vectors:
		pushed_up_dict[sample] = L2U.push_up(sample_vectors[sample], Tint, lint, nodes_in_order)
	return pushed_up_dict

def partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key):
	'''

	:param train_percentage: what percentage of the sample should be training samples.
	:param biom_file: a .biom file
	:param tree_file:
	:param metadata_file:
	:param metadata_key: The phenotype of interest. For example, 'body sites'
	:return:
	'''
	group_name_samples, sample_ids, classes = extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key)
	train_dict = {}
	test_dict = {}
	for c in classes:
		if c not in train_dict:
			train_dict[c] = {}
		if c not in test_dict:
			test_dict[c] = {}
		class_samples = [(key, value) for key, value in group_name_samples[c].items()]
		l = len(class_samples)
		train_num = floor(l*(train_percentage/100))
		test_num = l - train_num
		base_list = [0 for i in range(train_num)] + [1 for i in range(test_num)]
		for i in range(len(base_list)):
			if base_list[i] == 0:
				train_dict[c][class_samples[i][0]] = class_samples[i][1]
			if base_list[i] == 1:
				test_dict[c][class_samples[i][0]] = class_samples[i][1]

	return train_dict, test_dict

def combine_train_test(train_dict, test_dict):
	sample_dict = dict()
	for group in train_dict:
		sample_dict.update(train_dict[group])
	for group in test_dict:
		sample_dict.update(test_dict[group])
	return sample_dict


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.', nargs='?', default='data/metadata/P_1928_65684500_raw_meta.txt')
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?', default="body_site")
	parser.add_argument('-bf', '--biom_file', type=str, help='Path to the biom file.', nargs='?', default='data/biom/47422_otu_table.biom')
	parser.add_argument('-dm', '--distance_matrix', type=str, help="Pairwise unifrac distance matrix.", nargs='?', default='data/L1-UniFrac-Out.csv')
	parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
	parser.add_argument('-c', '--num_clusters', type=int, help="Number of clusters.", nargs='?', default=5)


	args = parser.parse_args()
	tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
	Tint, lint, nodes_in_order = parse_tree_file(tree_file)
	biom_file = args.biom_file
	metadata_file = args.meta_file
	metadata_key = args.phenotype
	distance_matrix = args.distance_matrix
	sample_ids = extract_samples(biom_file)
	meta_dict = extract_metadata(metadata_file)
	n_clusters = args.num_clusters

	train_dict, test_dict = partition_samples(80, biom_file, tree_file, metadata_file, metadata_key)
	sample_vector = combine_train_test(train_dict, test_dict)
	L2_vectors = push_up_all(sample_vector, Tint, lint, nodes_in_order)

	print("processed")
	km_score = get_KMedoids_clustering_score(distance_matrix, n_clusters, sample_ids, meta_dict)
	l2_score = get_L2_clustering_score(L2_vectors, n_clusters, meta_dict)
	print('KMedoids clustering score: ', km_score)
	print('L2UniFrac clustering score: ', l2_score)
