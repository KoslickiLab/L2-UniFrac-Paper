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
from sklearn.cluster import AgglomerativeClustering, KMeans
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
		print('len of vector:', len(vector))
		pushed_vector = L2U.push_up(vector, Tint, lint, nodes_in_order)
		#print("len of pushed vector:", len(pushed_vector))
		all_vectors.append(pushed_vector)
	mean_vector = L2U.mean_of_vectors(all_vectors)
	average_sample_vector = L2U.inverse_push_up(mean_vector, Tint, lint, nodes_in_order)
	#print(len(average_sample_vector))
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

def get_score_by_clustering_method(clustering_method, train_dict, test_dict, meta_dict, sample_dict, distance_matrix_file, n_clusters):
	distance_matrix = pd.read_csv(distance_matrix_file, header=None)
	#n_clusters = len(train_dict.keys()) #number of classes
	if clustering_method.lower() == "agglomerative": #case insensitive
		agglomerative_prediction = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete').fit_predict(distance_matrix)
		#accuracy score by body site
		results = get_clustering_scores(agglomerative_prediction, train_dict,test_dict, meta_dict, sample_dict)
	if clustering_method.lower() == "kmedoids":
		kmedoids_prediction = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='heuristic').fit_predict(distance_matrix)
		results = get_clustering_scores(kmedoids_prediction, train_dict, test_dict, meta_dict, sample_dict)
	return results

def get_clustering_scores(predictions, train_dict, test_dict, meta_dict, sample_dict):
	'''
	Returns a dict of scores for a particular set of predictions
	:param predictions:
	:param train_dict: {body_site:{sample_id: vector}}
	:param test_dict:
	:param meta_dict:
	:param sample_dict:
	:return:
	'''

	train_ids = get_sample_id_from_dict(train_dict)
	test_ids = get_sample_id_from_dict(test_dict)
	group_label_dict = dict()
	results_dict = dict()

	#decipher label
	index_sample_dict = {v:k for k,v in sample_dict.items()}
	for group in set(predictions):
		#label = decipher_label_by_vote(predictions, train_ids, group, meta_dict, sample_dict)
		label = decipher_label_alternative(predictions, index_sample_dict, group, meta_dict)
		group_label_dict[group] = label
	print(group_label_dict)
	for body_site in test_dict.keys():
		results_dict[body_site] = dict()
		test_ids_this_bs = test_dict[body_site]
		test_indices_this_bs = [sample_dict[sample_id] for sample_id in test_ids_this_bs]
		predicted_group_test_this_bs = [predictions[i] for i in test_indices_this_bs]
		predicted_labels_this_bs = [group_label_dict[group] for group in predicted_group_test_this_bs]
		true_labels_this_bs = [meta_dict[i]['body_site'] for i in test_ids_this_bs]
		results_dict[body_site]['accuracy_score'] = accuracy_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['rand_score'] = rand_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['adjusted_rand_score'] = adjusted_rand_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels_this_bs, predicted_labels_this_bs)
		results_dict[body_site]['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels_this_bs,predicted_labels_this_bs)
		results_dict[body_site]['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels_this_bs,predicted_labels_this_bs)
	test_indices = [sample_dict[sample_id] for sample_id in test_ids]
	predicted_group_test = [predictions[i] for i in test_indices]
	predicted_labels = [group_label_dict[group] for group in predicted_group_test]
	true_labels = [meta_dict[i]['body_site'] for i in test_ids]
	results_dict['overall'] = dict()
	results_dict['overall']['accuracy_score'] = accuracy_score(true_labels, predicted_labels)
	results_dict['overall']['rand_score'] = rand_score(true_labels, predicted_group_test)
	results_dict['overall']['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_group_test)
	results_dict['overall']['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels, predicted_group_test)
	results_dict['overall']['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predicted_group_test)
	results_dict['overall']['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels, predicted_group_test)
	print("clustering results:")
	print(results_dict)
	return results_dict

def get_sample_id_from_dict(t_dict):
	'''

	:param t_dict: {body_site:{sample_id:sample_vector}
	:return: a list of sample_id
	'''
	sample_lst = []
	for body_site in t_dict.keys():
		sample_lst+=list(t_dict[body_site].keys())
	return sample_lst

def get_L2UniFrac_accuracy_results(train_dict, test_dict,Tint, lint, nodes_in_order):
	results_dict = dict()
	rep_sample_dict = dict()
	for phenotype in train_dict.keys():
		print(phenotype)
		vectors = train_dict[phenotype].values()
		rep_sample = get_average_sample(vectors, Tint, lint, nodes_in_order)
		rep_sample_dict[phenotype] = rep_sample
	test_id = []
	overall_predictions = []
	all_true_labels = []
	for phenotype in test_dict.keys():
		test_id += list(test_dict[phenotype].keys())
		true_labels = [str(phenotype)] * len(test_dict[phenotype].keys())
		predictions = []
		for test_vector in test_dict[phenotype].values(): #list of samples in a particular phenotype
			prediction = get_label(test_vector, rep_sample_dict, Tint, lint, nodes_in_order)
			predictions.append(prediction)
		overall_predictions += predictions
		all_true_labels += true_labels
		results_dict[phenotype] = dict()
		results_dict[phenotype]['accuracy_score'] = accuracy_score(true_labels, predictions)
		results_dict[phenotype]['rand_score'] = rand_score(true_labels, predictions)
		results_dict[phenotype]['adjusted_rand_score'] = adjusted_rand_score(true_labels, predictions)
		results_dict[phenotype]['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels, predictions)
		results_dict[phenotype]['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predictions)
		results_dict[phenotype]['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels, predictions)
	results_dict['overall'] = dict()
	results_dict['overall']['accuracy_score'] = accuracy_score(all_true_labels, overall_predictions)
	results_dict['overall']['rand_score'] = rand_score(all_true_labels, overall_predictions)
	results_dict['overall']['adjusted_rand_score'] = adjusted_rand_score(all_true_labels, overall_predictions)
	results_dict['overall']['adjusted_mutual_info_score'] = adjusted_mutual_info_score(all_true_labels, overall_predictions)
	results_dict['overall']['normalized_mutual_info_score'] = normalized_mutual_info_score(all_true_labels, overall_predictions)
	results_dict['overall']['fowlkes_mallows_score'] = fowlkes_mallows_score(all_true_labels, overall_predictions)
	return results_dict

def compile_dataframe(n_repeat, train_percentage, biom_file, tree_file, metadata_file,
					  metadata_key, sample_dict, dm_file, n_clusters):

	col_names = ["Method", "Site", "Score_type", "Score"]
	df = pd.DataFrame(columns=col_names)
	score_type_col = []
	score_col = []
	site_col = []
	method_col = []
	for i in range(n_repeat):

		#agglomerative clustering
		#results = get_score_by_clustering_method("agglomerative", train_dict, test_dict, meta_dict, sample_dict, dm_file, n_clusters)
		#for site in results.keys(): #skin, gut, overall ...
		#	for score_type in results[site].keys():
		#		method_col.append("Agglomerative")
		#		site_col.append(site)
		#		score_type_col.append(score_type)
		#		score_col.append(results[site][score_type])
		train_dict, test_dict = partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)
		#KMeans
		all_vectors = []
		for body_site in train_dict.keys():
			for sample in train_dict[body_site].keys():
				all_vectors.append(train_dict[body_site][sample])
		for body_site in test_dict.keys():
			for sample in test_dict[body_site].keys():
				all_vectors.append(test_dict[body_site][sample])
		kmeans_predict = KMeans(n_clusters=n_clusters).fit_predict(all_vectors)
		results = get_clustering_scores(kmeans_predict, train_dict, test_dict, meta_dict, sample_dict)
		for site in results.keys(): #skin, gut, overall ...
			for score_type in results[site].keys():
				method_col.append("KMeans")
				site_col.append(site)
				score_type_col.append(score_type)
				score_col.append(results[site][score_type])
		# kmedoids clustering
		results = get_score_by_clustering_method("kmedoids", train_dict, test_dict, meta_dict, sample_dict, dm_file,
												 n_clusters)
		for site in results.keys():  # skin, gut, overall ...
			for score_type in results[site].keys():
				method_col.append("KMedoids")
				site_col.append(site)
				score_type_col.append(score_type)
				score_col.append(results[site][score_type])
		#L2UniFrac
		results = get_L2UniFrac_accuracy_results(train_dict,test_dict, Tint, lint, nodes_in_order)
		for site in results.keys(): #skin, gut, overall ...
			for score_type in results[site].keys():
				method_col.append("L2UniFrac")
				site_col.append(site)
				score_type_col.append(score_type)
				score_col.append(results[site][score_type])
	df["Method"] = method_col
	df["Site"] = site_col
	df["Score_type"] = score_type_col
	df["Score"] = score_col
	return df

def decipher_label_by_vote(predictions, training, group_name, meta_dict, sample_dict):
	'''
	:param predictions: a list of prediction of all samples. e.g. [0,1,3,0,...]
	:param training: list of training ids.
	:param meta_dict: body_site:sample_id dict
	:param group_name: cluster name. e.g. 0,1,2 ...
	:return: predicted label by vote
	'''
	train_id_this_group = [train_id for train_id in training if predictions[sample_dict[train_id]] == group_name]
	predicted_labels = [meta_dict[i]['body_site'] for i in train_id_this_group]
	#print(predicted_labels)
	c = Counter(predicted_labels)
	predicted_by_vote = c.most_common(1)[0][0]
	return predicted_by_vote

def decipher_label_alternative(predictions, index_sample_dict, group_name, meta_dict):
	'''
	A slight variation from the above function. The above function uses the overlap between training samples and each cluster to vote.
	Alternatively, use 80% data of each cluster to vote. Though we suspect the difference will not be significant.
	:return:
	'''
	indices_this_group = [i for i in range(len(predictions)) if predictions[i] == group_name]
	predicted_labels = [meta_dict[index_sample_dict[i]]['body_site'] for i in indices_this_group]
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


#extract_samples_by_group(biom_file, metadata_file, metadata_key)
#extract_sample_names_by_group(biom_file, metadata_file, metadata_key)
#extract_samples_direct(biom_file, tree_file)
#extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key)
#partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.', nargs='?', default='data/metadata/P_1928_65684500_raw_meta.txt')
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?', default="body_site")
	parser.add_argument('-bf', '--biom_file', type=str, help='Path to the biom file.', nargs='?', default='data/biom/47422_otu_table.biom')
	parser.add_argument('-tp', '--train_percentage', type=int, help='What percentage of data used in training.', nargs='?', default=80)
	parser.add_argument('-dm', '--distance_matrix', type=str, help="Pairwise unifrac distance matrix.", nargs='?', default='data/L2-UniFrac-Out.csv')
	parser.add_argument('-n', '--num_repeats', type=int, help="Number of repeats for each experiment.", nargs='?', default=10)
	parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
	parser.add_argument('-c', '--num_clusters', type=int, help="Number of clusters.", nargs='?', default=5)


	args = parser.parse_args()
	tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
	Tint, lint, nodes_in_order = parse_tree_file(tree_file)
	biom_file = args.biom_file
	metadata_file = args.meta_file
	metadata_key = args.phenotype
	train_percentage = args.train_percentage
	distance_matrix = args.distance_matrix
	sample_id = extract_samples(biom_file)
	sample_dict = get_index_dict(sample_id)
	meta_dict = extract_metadata(metadata_file)
	n_repeat = args.num_repeats
	n_clusters = args.num_clusters
	df = compile_dataframe(n_repeat, train_percentage, biom_file, tree_file, metadata_file, metadata_key, sample_dict, distance_matrix, n_clusters)
	print(df)
	df.to_csv(args.save, sep="\t")
