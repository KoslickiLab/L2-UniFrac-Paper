import os
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
	pheno_sample_dict = dict()
	for i, pheno in enumerate(targets_train):
		if pheno in pheno_sample_dict:
			pheno_sample_dict[pheno].append(samples_path_train[i])
		else:
			pheno_sample_dict[pheno] = [samples_path_train[i]]
	return pheno_sample_dict

def get_rep_sample_dict(pheno_sample_dict, Tint, lint, nodes_in_order, nodes_to_index):
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
	for group in set(predictions):
		label = decipher_label_by_vote(predictions, train_ids, group, meta_dict, sample_dict)
		# may need a tie breaker to ensure values are unique. For now just hope for the best
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
	#results_dict['overall']['accuracy_score'] = accuracy_score(true_labels, predicted_labels)
	#results_dict['overall']['rand_score'] = rand_score(true_labels, predicted_labels)
	#results_dict['overall']['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
	#results_dict['overall']['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels,predicted_labels)
	#results_dict['overall']['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predicted_labels)
	#results_dict['overall']['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels, predicted_labels)
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

def get_L2UniFrac_accuracy_results(train_dict, test_dict,Tint, lint, nodes_in_order, meta_dict):
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

def compile_dataframe(n_repeat, train_percentage, biom_file, tree_file, metadata_file, metadata_key, sample_dict, dm_file, n_clusters):

	col_names = ["Method", "Site", "Score_type", "Score"]
	df = pd.DataFrame(columns=col_names)
	score_type_col = []
	score_col = []
	site_col = []
	method_col = []
	for i in range(n_repeat):
		train_dict, test_dict = partition_sample(train_percentage, biom_file, tree_file, metadata_file, metadata_key)
		# kmedoids clustering
		results = get_score_by_clustering_method("kmedoids", train_dict, test_dict, meta_dict, sample_dict, dm_file,
												 n_clusters)
		for site in results.keys():  # skin, gut, overall ...
			for score_type in results[site].keys():
				method_col.append("KMedoids")
				site_col.append(site)
				score_type_col.append(score_type)
				score_col.append(results[site][score_type])
		#agglomerative clustering
		#results = get_score_by_clustering_method("agglomerative", train_dict, test_dict, meta_dict, sample_dict, dm_file, n_clusters)
		#for site in results.keys(): #skin, gut, overall ...
		#	for score_type in results[site].keys():
		#		method_col.append("Agglomerative")
		#		site_col.append(site)
		#		score_type_col.append(score_type)
		#		score_col.append(results[site][score_type])
		#L2UniFrac
		results = get_L2UniFrac_accuracy_results(train_dict,test_dict, Tint, lint, nodes_in_order, meta_dict)
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

def decipher_label_alternative():
	'''
	A slight variation from the above function. The above function uses the overlap between training samples and each cluster to vote.
	Alternatively, use 80% data of each cluster to vote. Though we suspect the difference will not be significant.
	:return:
	'''
	return

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
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?', default="HMgDB_diagnosis")
	#parser.add_argument('-t', '--test_size', type=int, help='What percentage of data used in testing.', nargs='?', default=0.2)
	parser.add_argument('-n', '--num_repeats', type=int, help="Number of repeats for each experiment.", nargs='?', default=10)
	parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
	parser.add_argument('-c', '--num_clusters', type=int, help="Number of clusters.", nargs='?', default=5)
	parser.add_argument('-d', '--pdir', type=str, help="Directory of profiles")

	args = parser.parse_args()
	metadata_file = args.meta_file
	metadata_key = args.phenotype
	#test_size = args.test_size
	n_repeat = args.num_repeats
	n_clusters = args.num_clusters
	profile_dir = args.pdir

	profile_path_lst = [os.path.join(profile_dir, file) for file in os.listdir(profile_dir)]
	Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_lst)
	meta_dict = get_metadata_dict(metadata_file, val_col=metadata_key)

	test_sizes = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

	col_names = ["Method", "Test_size", "Score_type", "Score"]
	df = pd.DataFrame(columns=col_names)
	scores_col = []
	test_size_col = []

	for test_size in test_sizes:
		print('test size:', test_size)
		for i in range(10):
			test_size_col.append(test_size)
			samples_train, samples_test, targets_train, targets_test = partition_sample(meta_dict, random_state=i, test_size=test_size)
			samples_train_paths = [profile_dir + '/' + sample + '.profile' for sample in samples_train]
			pheno_sample_dict = get_pheno_sample_dict(samples_train_paths, targets_train)
			rep_sample_dict = get_rep_sample_dict(pheno_sample_dict, Tint, lint, nodes_in_order, nodes_to_index) #get rep sample by phenotype
			#print(rep_sample_dict)
			samples_test_paths = [profile_dir + '/' + sample + '.profile' for sample in samples_test]
			test_sample_dict = L2U.merge_profiles_by_dir(samples_test_paths, nodes_to_index)
			prediction = [""] * len(targets_test)
			for i, sample in enumerate(samples_test):
				prediction[i] = get_label(test_sample_dict[sample], rep_sample_dict, Tint, lint, nodes_in_order)
			#print(prediction)
			accuracy = accuracy_score(prediction, targets_test)
			print(accuracy)
			scores_col.append(accuracy)
	df['Method'] = ['L2UniFrac'] * len(scores_col)
	df['Test_size'] = test_size_col
	df['Score_type'] = ['Accuracy'] * len(scores_col)
	df['Score'] = scores_col

	df.to_csv(args.save, sep="\t")







