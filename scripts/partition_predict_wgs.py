import copy
import os
import sys, biom
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
import argparse
import L2UniFrac as L2U
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score, rand_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score
from collections import Counter
from sklearn.model_selection import train_test_split


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

#functions needed

# 3. get_average_sample(profile_list, profile_dir) produces the representative sample given by profiles in profile_lst, which
# are present in profile_dir

def get_metadata_dict(meta_file, val_col = "HMgDB_diagnosis", key_col = "library_id"):
	meta_dict = dict()
	df = pd.read_csv(meta_file)
	for i, id in enumerate(df[key_col]):
		meta_dict[id] = df[val_col][i]
	return meta_dict

def partition_sample(meta_dict, random_state, test_size=0.2):
	'''
	Partitions samples in the meta_dict into training and testing sets
	:param meta_dict:
	:param percentage: percentage of training data.
	:return: train_dict, test_dict, {phenotype: [profile IDs]}
	'''
	sample_id = list(meta_dict.keys())
	targets = list(meta_dict.values()) #true phenotypes
	samples_train, samples_test, targets_train, targets_test = train_test_split(sample_id, targets, test_size=test_size, random_state=random_state)
	return samples_train, samples_test, targets_train, targets_test

def get_pheno_sample_dict(samples_path_train, targets_train):
	'''

	:param samples_path_train:
	:param targets_train:
	:return: {phenotype:[list of samples]}
	'''
	pheno_sample_dict = dict()
	for i, pheno in enumerate(targets_train):
		if pheno in pheno_sample_dict:
			pheno_sample_dict[pheno].append(samples_path_train[i])
		else:
			pheno_sample_dict[pheno] = [samples_path_train[i]]
	return pheno_sample_dict

def get_rep_sample_dict(pheno_sample_dict, Tint, lint, nodes_in_order, nodes_to_index):
	'''

	:param pheno_sample_dict:
	:param Tint:
	:param lint:
	:param nodes_in_order:
	:param nodes_to_index:
	:return: {phenotype:rep_sample}
	'''
	rep_sample_dict = dict()
	for pheno in pheno_sample_dict.keys():
		profile_path_list = pheno_sample_dict[pheno]
		rep_sample = L2U.get_representative_sample_wgs(profile_path_list, Tint, lint, nodes_in_order, nodes_to_index)
		rep_sample_dict[pheno] = rep_sample
	return rep_sample_dict

def get_label(test_vector, rep_sample_dict, Tint, lint, nodes_in_order):
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
		L2unifrac = L2U.L2UniFrac_weighted_plain(Tint, lint, nodes_in_order, test_vector, rep_sample_dict[phenotype])
		if L2unifrac < min_unifrac:
			min_unifrac = L2unifrac
			label = phenotype
	return label

def merge_clusters(predictions, group_label_dict):
	'''
	Merge predictions by collapsing clusters with the same phenotype
	:param predictions:
	:param group_label_dict: {group:label}
	:return:
	'''
	label_group_dict = get_reverse_dict(group_label_dict)
	merged_predictions = [label_group_dict[group_label_dict[i]] for i in predictions]
	updated_group_label_dict = get_reverse_dict(get_reverse_dict(group_label_dict))
	return merged_predictions, updated_group_label_dict

def try_cluster(init_n, max_try, true_n, clustering_method, clustering_basis, meta_dict):
	'''

	:param init_n: initial clustering number
	:param max_try: max number of clustering number to try
	:param true_n: true desired number of unique labels
	:param clustering_method: kmedoids or kmeans
	:param clustering_basis: distance matrix or sample_vector_dict, depending on clustering method
	:param sample_index_dict:
	:return:
	'''
	if clustering_method.lower() == "kmedoids":
		prediction, sample_ids = get_KMedoids_prediction(clustering_basis, init_n)
		print(prediction)
		print(type(prediction))
		sample_index_dict = get_index_dict(sample_ids)
		group_label_dict = get_group_label_dict(prediction, sample_index_dict, meta_dict)
		while len(set(group_label_dict.values())) < true_n and init_n < max_try:
			init_n+=1
			prediction, sample_ids = get_KMedoids_prediction(clustering_basis, init_n)
			group_label_dict = get_group_label_dict(prediction, sample_index_dict, meta_dict)
		if len(set(group_label_dict.values())) < true_n:
			print("Clustering results still not ideal but I did my best. Try increasing max_n")
		else:
			print("After trying {} times it worked".format(init_n))
	elif clustering_method.lower() == "kmeans":
		prediction, sample_ids = get_KMeans_prediction(clustering_basis, init_n) #clustering_basis = sample:vector dict
		sample_index_dict = get_index_dict(list(clustering_basis.keys()))
		group_label_dict = get_group_label_dict(prediction, sample_index_dict, meta_dict)
		while len(set(group_label_dict.values())) < true_n and init_n < max_try:
			init_n+=1
			prediction, sample_ids = get_KMeans_prediction(clustering_basis,
													  init_n)  # clustering_basis = sample:vector dict
			group_label_dict = get_group_label_dict(prediction, sample_index_dict, meta_dict)
		if len(set(group_label_dict.values())) < true_n:
			print("Clustering results still not ideal but I did my best. Try increasing max_n")
		else:
			print("After trying {} times it worked".format(init_n))
	else:
		prediction = 0
		group_label_dict = 0
		sample_ids = 0
		print("Clustering method is wrong")
	merged_prediction, updated_group_label_dict = merge_clusters(prediction, group_label_dict)
	return merged_prediction, updated_group_label_dict, sample_ids

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
	predicted_labels = [meta_dict[index_sample_dict[i]] for i in indices_this_group]
	c = Counter(predicted_labels)
	predicted_by_vote = c.most_common(1)[0][0]
	return predicted_by_vote

def get_group_label_dict(predictions, sample_index_dict, meta_dict):
	group_label_dict = dict()
	index_sample_dict = get_reverse_dict(sample_index_dict)
	# decipher label
	#print(predictions)
	for group in set(predictions):
		label = decipher_label_alternative(predictions, index_sample_dict, group, meta_dict)
		# may need a tie breaker to ensure values are unique. For now just hope for the best
		group_label_dict[group] = label
	#print('group label dict:')
	#print(group_label_dict)
	return group_label_dict

def get_clustering_scores(predictions, test_ids, meta_dict, sample_index_dict, group_label_dict):
	'''
	Returns a dict of scores for a particular set of predictions
	:param predictions:
	:param train_dict:
	:param test_dict:
	:param meta_dict:
	:param sample_dict:
	:return:
	'''
	results_dict = dict()
	test_indices = [sample_index_dict[sample_id] for sample_id in test_ids]
	predicted_group_test = [predictions[i] for i in test_indices]
	predicted_labels = [group_label_dict[group] for group in predicted_group_test]
	true_labels = [meta_dict[i] for i in test_ids]
	results_dict['accuracy_score'] = accuracy_score(true_labels, predicted_labels)
	results_dict['balanced_accuracy_score'] = balanced_accuracy_score(true_labels, predicted_labels)
	results_dict['rand_score'] = rand_score(true_labels, predicted_group_test)
	results_dict['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_group_test)
	results_dict['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels, predicted_group_test)
	results_dict['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predicted_group_test)
	results_dict['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels, predicted_group_test)
	results_dict['precision_micro'] = precision_score(true_labels, predicted_labels, average='micro')
	results_dict['precision_macro'] = precision_score(true_labels, predicted_labels, average='macro')
	results_dict['recall_micro'] = recall_score(true_labels, predicted_labels, average='micro')
	results_dict['recall_macro'] = recall_score(true_labels, predicted_labels, average='macro')
	#print("clustering results:")
	#print(results_dict)
	return results_dict

def get_KMedoids_prediction(dmatrix_file, n_clusters):
	distance_matrix = pd.read_csv(dmatrix_file, header=0, index_col=0, sep='\t')
	sample_ids =  distance_matrix.columns
	kmedoids_prediction = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam',
								   init='heuristic').fit_predict(distance_matrix)
	return kmedoids_prediction, sample_ids

def get_KMeans_prediction(sample_vector_dict, n_clusters):
	sample_ids = list(sample_vector_dict.keys())
	all_vectors = [sample_vector_dict[i] for i in sample_ids]
	kmeans_predict = KMeans(n_clusters=n_clusters).fit_predict(all_vectors)
	return kmeans_predict, sample_ids

def get_sample_id_from_dict(t_dict):
	'''

	:param t_dict: {body_site:{sample_id:sample_vector}
	:return: a list of sample_id
	'''
	sample_lst = []
	for phenotype in t_dict.keys():
		sample_lst+=list(t_dict[phenotype].keys())
	return sample_lst

def get_L2UniFrac_accuracy_results(test_ids, test_targets,Tint, lint, nodes_in_order, rep_sample_dict, sample_vector_dict):
	results_dict = dict()
	overall_predictions = []
	for i, id in enumerate(test_ids):
		prediction = get_label(sample_vector_dict[id], rep_sample_dict, Tint, lint, nodes_in_order)
		overall_predictions.append(prediction)

	results_dict['accuracy_score'] = accuracy_score(test_targets, overall_predictions)
	results_dict['rand_score'] = rand_score(test_targets, overall_predictions)
	results_dict['adjusted_rand_score'] = adjusted_rand_score(test_targets, overall_predictions)
	results_dict['adjusted_mutual_info_score'] = adjusted_mutual_info_score(test_targets, overall_predictions)
	results_dict['normalized_mutual_info_score'] = normalized_mutual_info_score(test_targets, overall_predictions)
	results_dict['fowlkes_mallows_score'] = fowlkes_mallows_score(test_targets, overall_predictions)
	results_dict['balanced_accuracy_score'] = balanced_accuracy_score(test_targets, overall_predictions)
	results_dict['precision_micro'] = precision_score(test_targets, overall_predictions, average='micro')
	results_dict['precision_macro'] = precision_score(test_targets, overall_predictions, average='macro')
	results_dict['recall_micro'] = recall_score(test_targets, overall_predictions, average='micro')
	results_dict['recall_macro'] = recall_score(test_targets, overall_predictions, average='macro')
	return results_dict

def get_index_dict(lst):
	'''
	Get a dictionary with keys being items in the list and values being their respective positions in the list
	:param lst:
	:return:
	'''
	index_dict = dict(zip(lst, range(len(lst))))
	return index_dict

def get_reverse_dict(dct):
	reverse_dict = {x:y for y,x in dct.items()}
	return reverse_dict


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.', nargs='?')
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?', default="HMgDB_diagnosis")
	parser.add_argument('-n', '--num_repeats', type=int, help="Number of repeats for each experiment.", nargs='?', default=10)
	parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
	parser.add_argument('-dm', '--distance_matrix', type=str, help="Pairwise unifrac distance matrix.")
	parser.add_argument('-d', '--pdir', type=str, help="Directory of profiles")

	args = parser.parse_args()
	metadata_file = args.meta_file
	metadata_key = args.phenotype
	n_repeat = args.num_repeats
	profile_dir = args.pdir

	profile_path_lst = [os.path.join(profile_dir, file) for file in os.listdir(profile_dir)]
	Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_lst)
	meta_dict = get_metadata_dict(metadata_file, val_col=metadata_key)

	test_sizes = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

	col_names = ["Method", "Test_size", "accuracy_score", 'rand_score', 'adjusted_rand_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'fowlkes_mallows_score', 'balanced_accuracy_score', 'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro']
	df = pd.DataFrame(columns=col_names)
	method_col = []
	test_size_col = []

	#clusterings (independent of sample splitting)
	true_n = len(set(meta_dict))
	init_n = true_n
	#kmedoids
	kmedoids_prediction, kmedoids_group_label_dict, kmedoids_sample_ids = try_cluster(init_n, 30, true_n, "kmedoids", args.distance_matrix, meta_dict)
	kmedoids_sample_index_dict = get_index_dict(kmedoids_sample_ids)
	#kmeans
	all_samples = sample_id = list(meta_dict.keys())
	all_samples_paths = [profile_dir + '/' + sample + '.profile' for sample in all_samples]
	sample_vector_dict = L2U.merge_profiles_by_dir(all_samples_paths, nodes_to_index)
	init_n = true_n #may not be needed, just to be safe
	kmeans_prediction, kmeans_group_label_dict, kmeans_sample_ids = try_cluster(init_n, 5, true_n, 'kmeans', sample_vector_dict, meta_dict)
	kmeans_sample_index_dict = get_index_dict(kmeans_sample_ids)

	for test_size in test_sizes:
		print('test size:', test_size)
		for i in range(n_repeat):
			#data prep
			col_names = ["Method", "Test_size", "accuracy_score", 'rand_score', 'adjusted_rand_score',
						 'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'fowlkes_mallows_score',
						 'balanced_accuracy_score', 'precision_micro', 'precision_macro', 'recall_micro',
						 'recall_macro']
			df_tmp = pd.DataFrame(columns=col_names)
			df_tmp['Test_size'] = [test_size] * 3
			df_tmp['Method'] = ['L2UniFrac', 'KMedoids', 'KMeans']
			samples_train, samples_test, targets_train, targets_test = partition_sample(meta_dict, random_state=i, test_size=test_size)
			samples_train_paths = [profile_dir + '/' + sample + '.profile' for sample in samples_train]
			pheno_sample_dict = get_pheno_sample_dict(samples_train_paths, targets_train)
			rep_sample_dict = get_rep_sample_dict(pheno_sample_dict, Tint, lint, nodes_in_order, nodes_to_index) #get rep sample by phenotype
			#L2UniFrac results
			L2_results_dict = get_L2UniFrac_accuracy_results(samples_test, targets_test, Tint, lint, nodes_in_order, rep_sample_dict, sample_vector_dict)
			#KMedoids results
			kmedoids_results = get_clustering_scores(kmedoids_prediction, samples_test, meta_dict, kmedoids_sample_index_dict, kmedoids_group_label_dict)
			#KMeans results
			kmeans_results = get_clustering_scores(kmeans_prediction, samples_test, meta_dict, kmeans_sample_index_dict, kmeans_group_label_dict)
			for score_type in L2_results_dict.keys():
				col_tmp = []
				col_tmp.append(L2_results_dict[score_type])
				col_tmp.append(kmedoids_results[score_type])
				col_tmp.append(kmeans_results[score_type])
				df_tmp[score_type] = col_tmp
			df = pd.concat([df, df_tmp], ignore_index=True)
	df.to_csv(args.save, sep="\t")







