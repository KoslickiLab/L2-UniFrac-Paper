import sys
import time

sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import argparse
import numpy as np
from math import floor
import L2UniFrac as L2U
import L1UniFrac as L1U
from extract_data import extract_biom, extract_samples, extract_metadata, parse_tree_file, parse_envs, extract_biom_samples
import argparse
import pandas as pd
from sklearn.metrics import fowlkes_mallows_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import train_test_split

#time the following: 1. producing L1 UniFrac pairwise distance matrix and clustering using KMedoids. 2. Push up vectors to L2 space, and cluster.
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

def partition_sample(meta_dict, sample_dict, test_size=0.2):
	'''
	Partitions samples in the meta_dict into training and testing sets. Use as the sampling function.
	:param meta_dict:
	:param percentage: percentage of training data.
	:return: train_dict, test_dict, {phenotype: [profile IDs]}
	'''
	sample_id = list(sample_dict.keys())
	targets = [meta_dict[i] for i in sample_id]
	samples_train, samples_test, targets_train, targets_test = train_test_split(sample_id, targets, test_size=test_size)
	return samples_test, targets_test

def get_L2UniFrac_method_time(sample_ids, meta_dict, sample_dict, Tint, lint, nodes_in_order):
	start_time = time.time()
	#Kmeans on pushed-up vectors
	pushed_sample_dict = push_up_by_id(sample_ids, sample_dict, Tint, lint, nodes_in_order)
	kmeans_predict = KMeans(n_clusters=5).fit_predict(list(pushed_sample_dict.values()))
	labels = get_true_label(meta_dict, sample_ids)
	return time.time() - start_time, fowlkes_mallows_score(kmeans_predict, labels)

def get_true_label(meta_dict, sample_ids):
	'''

	:param meta_dict: {sample_id: 'body_site': body_site}
	:param sample_ids: list of sample ids found in meta_dict
	:return: a list of true labels
	'''
	true_labels = [meta_dict[i]['body_site'] for i in sample_ids]
	return true_labels

def push_up_by_id(sample_ids, sample_dict, Tint, lint, nodes_in_order):
	'''
	Takes in a list of ids and returns a dict of ids:pushed_up_vector
	:param sample_ids:
	:return:
	'''
	pushed_dict = dict()
	print(sample_ids)
	for sample in sample_ids:
		print(sample)
		pushed_vector = L2U.push_up(sample_dict[sample], Tint, lint, nodes_in_order)
		pushed_dict[sample] = pushed_vector
	return pushed_dict

def get_traditional_method_time(sample_ids, sample_dict, meta_dict, Tint, lint, nodes_in_order):
	start_time = time.time()
	distance_matrix = L1U.pairwise_L1EMDUniFrac_weighted(sample_dict, Tint, lint, nodes_in_order)
	print(distance_matrix)
	labels = get_true_label(meta_dict, sample_ids)
	kmedoids_prediction = KMedoids(n_clusters=5, metric='precomputed', method='pam', init='heuristic').fit_predict(distance_matrix)
	return time.time() - start_time, fowlkes_mallows_score(kmedoids_prediction, labels)

def compile_dataframe(meta_dict, sample_dict, Tint, lint, nodes_in_order, save_as):
	total_size = len(meta_dict)
	col_names = ["Method", "Sample_size", "Time", "Fowlkes_Mallows_score"]
	time_col = []
	score_col = []
	df = pd.DataFrame(columns=col_names)
	sample_size = np.arange(0.1, 1, 0.1)
	method_col = ['L2UniFrac', 'Matrix-based'] * len(sample_size) * 10
	size_col = []
	for i in range(10):
		for size in sample_size:
			size_col+=[size * total_size]*2
			sample_ids, sample_targets = partition_sample(meta_dict, test_size=size)
			t, s = get_L2UniFrac_method_time(sample_ids, meta_dict, sample_dict, Tint, lint, nodes_in_order)
			time_col.append(t)
			score_col.append(s)
			t = get_traditional_method_time()
			time_col.append(t)
			score_col.append(s)
	df['Method'] = method_col
	df['Sample_size'] = size_col
	df['Time'] = time_col
	df["Fowlkes_Mallows_score"] = score_col
	df.to_csv(save_as, sep='\t')
	return

def partition_samples_to_dict(train_percentage, biom_file, tree_file, metadata_file, metadata_key):
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
	parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
	parser.add_argument('-c', '--num_clusters', type=int, help="Number of clusters.", nargs='?', default=5)


	args = parser.parse_args()
	tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
	Tint, lint, nodes_in_order = parse_tree_file(tree_file)
	biom_file = args.biom_file
	metadata_file = args.meta_file
	metadata_key = args.phenotype
	sample_ids = extract_samples(biom_file)
	meta_dict = extract_metadata(metadata_file)
	n_clusters = args.num_clusters
	save_as = args.save

	train_dict, test_dict = partition_samples_to_dict(80, biom_file, tree_file, metadata_file, metadata_key)
	sample_vector = combine_train_test(train_dict, test_dict)
	print(list(sample_vector.keys())[0:2])
	print(len(sample_vector))
	compile_dataframe(meta_dict, sample_vector, Tint, lint, nodes_in_order, save_as)