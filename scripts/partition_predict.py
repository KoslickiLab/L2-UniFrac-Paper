import sys, biom
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
from extract_data import extract_biom, extract_samples, extract_metadata, parse_tree_file, parse_envs
from random import shuffle
from math import floor
import argparse
import L2UniFrac as L2U
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score, rand_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score
from collections import Counter


class TrainingRateTooHighOrLow(Exception):

    def __init__(self, train_percentage, msg="Invalid training percentage. Percentage should be between 10 and 90."):
        self.train_percentage = train_percentage
        self.msg = msg
        super().__init__(self.msg)

class ClassTooSmall(Exception):

    def __init__(self, c, msg="Class size too small. Minimum size is 100 samples."):
        self.c = c
        self.msg = msg
        super().__init__(self.msg)

def extract_samples_by_group(biom_file, metadata_file, metadata_key):
	'''

	:param biom_file: A .biom file
	:param metadata_file: Metadata file
	:param metadata_key: The phenotype of interest. For example, 'body sites'
	:return:
	'''
	Biom = biom.load_table(biom_file)
	sample_ids = Biom.ids()
	tree_nodes = Biom.ids(axis='observation')
	metadata = extract_metadata(metadata_file)
	nodes_test = extract_biom(biom_file)
	group_samples = {}
	for sample in sample_ids:
		col = []
		for node in tree_nodes:
			col.append(nodes_test[node][sample])
		if metadata[sample][metadata_key] not in group_samples:
			group_samples[metadata[sample][metadata_key]] = []
		group_samples[metadata[sample][metadata_key]].append(col)

	return group_samples

def extract_sample_names_by_group(biom_file, metadata_file, metadata_key):
	Biom = biom.load_table(biom_file)
	sample_ids = Biom.ids()
	metadata = extract_metadata(metadata_file)
	group_name_samples = {}
	for sample in sample_ids:
		if metadata[sample][metadata_key] not in group_name_samples:
			group_name_samples[metadata[sample][metadata_key]] = []
		group_name_samples[metadata[sample][metadata_key]].append(sample)

	return group_name_samples

def extract_samples_direct(biom_file, tree_file):
	nodes_samples = extract_biom(biom_file)
	_, _, nodes_in_order = parse_tree_file(tree_file)
	(nodes_weighted, samples_temp) = parse_envs(nodes_samples, nodes_in_order)
	sample_ids = extract_samples(biom_file)

	return nodes_weighted, sample_ids

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

# Partition between train and test randomly
def partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key):
	'''

	:param train_percentage: what percentage of the sample should be training samples.
	:param biom_file: a .biom file
	:param tree_file:
	:param metadata_file:
	:param metadata_key: The phenotype of interest. For example, 'body sites'
	:return:
	'''
	try:
		assert train_percentage <= 90 and train_percentage >= 10
	except:
		raise TrainingRateTooHighOrLow(train_percentage)

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
		try:
			assert l > 100
		except:
			raise ClassTooSmall(c)
		train_num = floor(l*(train_percentage/100))
		test_num = l - train_num
		base_list = [0 for i in range(train_num)] + [1 for i in range(test_num)]
		shuffled_list = shuffle(base_list)
		for i in range(len(base_list)):
			if base_list[i] == 0:
				train_dict[c][class_samples[i][0]] = class_samples[i][1]
			if base_list[i] == 1:
				test_dict[c][class_samples[i][0]] = class_samples[i][1]

	return train_dict, test_dict

def get_average_sample(sample_list, Tint, lint, nodes_in_order):
	'''

	:param sample_list: a list of vectors of which the average sample vector is to be computed
	:return: average vector
	'''
	all_vectors = []
	for vector in sample_list:
		pushed_vector = L2U.push_up(vector, Tint, lint, nodes_in_order)
		all_vectors.append(pushed_vector)
	mean_vector = L2U.mean_of_vectors(all_vectors)
	average_sample_vector = L2U.inverse_push_up(mean_vector, Tint, lint, nodes_in_order)
	return average_sample_vector

def get_label(test_sample, rep_sample_dict, Tint, lint, nodes_in_order):
	'''
	predict label by min unifrac
	:param test_sample:
	:param rep_sample_dict:
	:param Tint:
	:param lint:
	:param nodes_in_order:
	:return:
	'''
	min_unifrac = 1000
	label = ""
	for phenotype in rep_sample_dict:
		L2unifrac = L2U.L2UniFrac_weighted_plain(Tint, lint, nodes_in_order, test_sample, rep_sample_dict[phenotype])
		if L2unifrac < min_unifrac:
			min_unifrac = L2unifrac
			label = phenotype
	return label

def get_score_by_clustering_method(clustering_method, train_dict, test_dict, meta_dict, sample_dict):
	pd.read_csv(distance_matrix, header=None)
	n_clusters = len(train_dict.keys()) #number of classes
	if clustering_method.lower() == "agglomerative": #case insensitive
		agglomerative_prediction = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete').fit_predict(distance_matrix)
		#accuracy score by body site
		results = get_clustering_scores(agglomerative_prediction, train_dict,test_dict, meta_dict, sample_dict)
	if clustering_method.lower() == "kmedoids":
		kmedois_prediction = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='heuristic').fit_predict(distance_matrix)
		results = get_clustering_scores(kmedois_prediction, train_dict, test_dict, meta_dict, sample_dict)


def get_clustering_scores(predictions, train_dict, test_dict, meta_dict, sample_dict):
	'''
	Returns a dict of scores for a particular set of predictions
	:param predictions:
	:param train_dict:
	:param test_dict:
	:param meta_dict:
	:param sample_dict:
	:return:
	'''
	train_ids = train_dict.values()
	test_ids = test_dict.values()
	group_label_dict = dict()
	results_dict = dict()
	#decipher label
	for group in set(predictions):
		label = decipher_label_by_vote(predictions, train_ids, group, meta_dict, sample_dict)
		# may need a tie breaker to ensure values are unique
		group_label_dict[group] = label
	for body_site in test_dict.keys():
		test_ids_this_bs = test_dict[body_site]
		test_indices_this_bs = [sample_dict[sample_id] for sample_id in test_ids_this_bs]
		predicted_group_test_this_bs = [predictions[i] for i in test_indices_this_bs]
		predicted_labels_this_bs = [group_label_dict[group] for group in predicted_group_test_this_bs]
		true_labels_this_bs = [meta_dict[i]['body_site'] for i in test_indices_this_bs]
		results_dict[body_site]['accuracy_score'] = accuracy_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['rand_score'] = rand_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['adjusted_rand_score'] = adjusted_rand_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels_this_bs,predicted_labels_this_bs)
		results_dict[body_site]['fawlkes_mallows_score'] = fowlkes_mallows_score(true_labels_this_bs,predicted_labels_this_bs)
	test_indices = [sample_dict[sample_id] for sample_id in test_ids]
	predicted_group_test = [predictions[i] for i in test_indices]
	predicted_labels = [group_label_dict[group] for group in predicted_group_test]
	true_labels = [meta_dict[i]['body_site'] for i in test_ids]
	results_dict['overall']['accuracy_score'] = accuracy_score(true_labels, predicted_labels)
	results_dict['overall']['rand_score'] = rand_score(true_labels, predicted_labels)
	results_dict['overall']['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
	results_dict['overall']['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels,predicted_labels)
	results_dict['overall']['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predicted_labels)
	results_dict['overall']['fawlkes_mallows_score'] = fowlkes_mallows_score(true_labels, predicted_labels_this_bs)
	return results_dict

def compile_dataframe(n_repeat):
	col_names = ["Method", "Score type", "Score"]
	df_agg = pd.DataFrame(col_names=col_names)
	df_agg["Method"] = "Agglomerative"
	for i in range(n_repeat):
		train_dict, test_dict = partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)




def decipher_label_by_vote(predictions, training, group_name, meta_dict, sample_dict):
	'''
	:param predictions: a list of prediction of all samples. e.g. [0,1,3,0,...]
	:param training: list of training ids.
	:param meta_dict: body_site:sample_id dict
	:param group_name: cluster name. e.g. 0,1,2 ...
	:return: predicted label by vote
	'''
	train_id_this_group = [train_id for train_id in training if predictions[sample_dict[train_id]] == group_name]
	print(train_id_this_group)
	predicted_labels = [meta_dict[i]['body_site'] for i in train_id_this_group]
	print(predicted_labels)
	c = Counter(predicted_labels)
	predicted_by_vote = c.most_common(1)[0][0]
	return predicted_by_vote

def get_index_dict(lst):
	'''
	Get a dictionary with keys being items in the list and values being their respective positions in the list
	:param lst:
	:return:
	'''
	index_dict = dict(zip(lst, range(len(lst))))
	return index_dict

biom_file = 'data/biom/47422_otu_table.biom'
metadata_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
metadata_key = 'body_site'
train_percentage = 80
distance_matrix = 'data/L2-UniFrac-Out.csv'
n_clusters = 5 # 5 body sites
sample_id = extract_samples(biom_file)
sample_dict = get_index_dict(sample_id)
meta_dict = extract_metadata(metadata_file)
#extract_samples_by_group(biom_file, metadata_file, metadata_key)
#extract_sample_names_by_group(biom_file, metadata_file, metadata_key)
#extract_samples_direct(biom_file, tree_file)
#extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key)
#partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.')
	parser.add_argument('-d', '--dir', type=str, help='A directory containing profiles. Only required if data type is wgs.')
	parser.add_argument('-msg', '--message', type=str, help='Message printed in the output file before test statistics.')
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.')
	parser.add_argument('-dt', '--data_type', type=str, help='wgs or 16s.')
	parser.add_argument('-ot', '--otu_table', type=str, help='Path to the otu table.')
	parser.add_argument('-t', '--tree', type=str, help='Path to tree file. Only needed if data type is 16s')
	parser.add_argument('-o', '--out_file', type=str, help='Path to the output file. Results will be printed in this file.')
	parser.add_argument('-tp', '--train_percentage', type=int, help='What percentage of data used in training.')

	args = parser.parse_args()
	#if args.data_type == '16s':
	Tint, lint, nodes_in_order = parse_tree_file(args.tree)
	rep_sample_dict = dict()
	print('preprocessing completed')
	test_results = dict()
	if args.data_type == '16s':
		train_dict, test_dict = partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)
		#get representative sample dict
		for phenotype in train_dict.keys():
			print(phenotype)
			vectors = train_dict[phenotype].values()
			rep_sample = get_average_sample(vectors, Tint, lint, nodes_in_order)
			rep_sample_dict[phenotype] = rep_sample
			print(rep_sample)
		#test
		total_test = 0
		total_correct = 0
		for phenotype in test_dict.keys():
			total_this_pheno = 0
			total_correct_this_pheno = 0
			for test_vector in test_dict[phenotype].values():
				prediction = get_label(test_vector, rep_sample_dict, Tint, lint, nodes_in_order)
				total_this_pheno+=1
				total_test+=1
				if prediction == phenotype:
					total_correct_this_pheno+=1
					total_correct+=1
			accuracy_this_pheno = total_correct_this_pheno/total_this_pheno
			test_results[phenotype] = accuracy_this_pheno
		total_acc = total_correct/total_test
		#print results
		print(total_acc)
		print(test_results)
		with open(args.out_file, 'a+') as f:
			f.write('#{}\n'.format(args.message))
			for phenotype in test_results.keys():
				f.write('{} : {}\n'.format(phenotype, test_results[phenotype]))
			f.write('Total accuracy : {}\n'.format(total_acc))


	#print(len(train_dict.keys()))
	#print(len(test_dict.keys()))

