import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import argparse
from parse_data import parse_df
from extract_data import parse_tree_file, extract_samples_direct
import L2UniFrac as L2U
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score, rand_score, precision_score, recall_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score
from helper import decipher_label_by_vote, get_label_by_proximity, partition_samples, get_metadata_dict, get_meta_samples_dict


def get_average_sample(sample_list, Tint, lint, nodes_in_order):
	'''

	:param sample_list: a list of vectors of which the average sample vector is to be computed
	:return: average vector
	'''
	all_vectors = []
	for vector in sample_list:
		pushed_vector = L2U.push_up(vector, Tint, lint, nodes_in_order)
		#print("len of pushed vector:", len(pushed_vector))
		all_vectors.append(pushed_vector)
	mean_vector = L2U.mean_of_vectors(all_vectors)
	average_sample_vector = L2U.inverse_push_up(mean_vector, Tint, lint, nodes_in_order)
	#print(len(average_sample_vector))
	return average_sample_vector

def get_score_by_clustering_method(clustering_method, test_ids, meta_dict, sample_ids, distance_matrix_file, n_clusters):
	distance_matrix = pd.read_csv(distance_matrix_file, header=None)
	#n_clusters = len(train_dict.keys()) #number of classes
	if clustering_method.lower() == "agglomerative": #case insensitive
		agglomerative_prediction = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete').fit_predict(distance_matrix)
		#accuracy score by body site
		results = get_clustering_scores(agglomerative_prediction, test_ids, meta_dict, sample_ids)
	if clustering_method.lower() == "kmedoids":
		kmedoids_prediction = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='heuristic').fit_predict(distance_matrix)
		results = get_clustering_scores(kmedoids_prediction, test_ids, meta_dict, sample_ids)
	return results

def get_clustering_scores(predictions, test_ids, meta_dict, sample_ids):
	'''
	Returns a dict of scores for a particular set of predictions
	:param predictions:
	:param train_dict: {body_site:{sample_id: vector}}
	:param test_dict:
	:param meta_dict:
	:param sample_dict:
	:return:
	'''
	group_label_dict = dict()
	#decipher label
	index_sample_dict = dict(zip(range(len(sample_ids)), sample_ids))
	sample_index_dict = dict(zip(sample_ids, range(len(sample_ids))))

	for group in set(predictions):
		label = decipher_label_by_vote(predictions, index_sample_dict, group, meta_dict)
		group_label_dict[group] = label
	print(group_label_dict)
	test_indices = [sample_index_dict[sample_id] for sample_id in test_ids]
	predicted_group_test = [predictions[i] for i in test_indices]
	predicted_labels = [group_label_dict[group] for group in predicted_group_test]
	true_labels = [meta_dict[i] for i in test_ids]
	results_dict = dict()
	print('overall prediction ', predicted_group_test[:5])
	print('test targets', true_labels[:5])
	results_dict['accuracy_score'] = accuracy_score(true_labels, predicted_labels)
	results_dict['rand_score'] = rand_score(true_labels, predicted_group_test)
	results_dict['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_group_test)
	results_dict['adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels, predicted_group_test)
	results_dict['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predicted_group_test)
	results_dict['fowlkes_mallows_score'] = fowlkes_mallows_score(true_labels, predicted_group_test)
	results_dict['precision_micro'] = precision_score(true_labels, predicted_group_test, average='micro')
	results_dict['precision_macro'] = precision_score(true_labels, predicted_group_test, average='macro')
	results_dict['recall_micro'] = recall_score(true_labels, predicted_group_test, average='micro')
	results_dict['recall_macro'] = recall_score(true_labels, predicted_group_test, average='macro')
	print("clustering results:")
	print(results_dict)
	return results_dict


def get_rep_sample_dict(meta_sample_dict, sample_vector_dict, Tint, lint, nodes_in_order):
	'''
	Given a dict of phenotype:sample_ids, compute the representative vector for each phenotype and return as a dict
	:param meta_sample_dict: {phenotype: [sample_ids]}
	:param Tint:
	:param lint:
	:param nodes_in_order:
	:param nodes_to_index:
	:return: {phenotype:rep_sample}
	'''
	rep_sample_dict = dict()
	for pheno in meta_sample_dict.keys():
		rep_sample = L2U.get_representative_sample_16s(sample_vector_dict, meta_sample_dict, Tint, lint, nodes_in_order)
		rep_sample_dict[pheno] = rep_sample
	return rep_sample_dict

def get_L2UniFrac_accuracy_results(test_ids, test_targets, Tint, lint, nodes_in_order, rep_sample_dict, sample_vector_dict):
	results_dict = dict()
	overall_predictions = []
	for id in test_ids:
		prediction = get_label_by_proximity(sample_vector_dict[id], rep_sample_dict, Tint, lint, nodes_in_order)
		overall_predictions.append(prediction)

	results_dict['accuracy_score'] = accuracy_score(test_targets, overall_predictions)
	results_dict['rand_score'] = rand_score(test_targets, overall_predictions)
	results_dict['adjusted_rand_score'] = adjusted_rand_score(test_targets, overall_predictions)
	results_dict['adjusted_mutual_info_score'] = adjusted_mutual_info_score(test_targets, overall_predictions)
	results_dict['normalized_mutual_info_score'] = normalized_mutual_info_score(test_targets, overall_predictions)
	results_dict['fowlkes_mallows_score'] = fowlkes_mallows_score(test_targets, overall_predictions)
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

def argument_parser():
	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.', nargs='?', default='data/metadata/P_1928_65684500_raw_meta.txt')
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?', default="body_site")
	parser.add_argument('-bf', '--biom_file', type=str, help='Path to the biom file.', nargs='?', default='data/1928_body_sites/47422_otu_table.biom')
	parser.add_argument('-dm', '--distance_matrix', type=str, help="Pairwise unifrac distance matrix.", nargs='?', default='data/L2-UniFrac-Out.csv')
	parser.add_argument('-n', '--num_repeats', type=int, help="Number of repeats for each experiment.", nargs='?', default=10)
	parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
	parser.add_argument('-c', '--num_clusters', type=int, help="Number of clusters.", nargs='?', default=5)
	return parser

def main():
	tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
	parser = argument_parser()
	args = parser.parse_args()
	Tint, lint, nodes_in_order = parse_tree_file(tree_file)
	sample_vector_dict, sample_ids = extract_samples_direct(args.biom_file, tree_file)
	meta_dict = get_metadata_dict(args.meta_file, val_col=args.phenotype, key_col="sample_name")
	meta_dict = {k:meta_dict[k] for k in sample_ids}
	meta_sample_dict = get_meta_samples_dict(meta_dict)
	print(meta_sample_dict.keys())
	#compile dataframe
	col_names = ["Method", "Score_type", "Score"]
	df = pd.DataFrame(columns=col_names)
	score_type_col = []
	score_col = []
	method_col = []
	for i in range(args.num_repeats):
		samples_train, samples_test, targets_train, targets_test = partition_samples(meta_dict, random_state=i)
		#KMeans
		all_vectors = list(sample_vector_dict.values())
		kmeans_predict = KMeans(n_clusters=args.num_clusters).fit_predict(all_vectors)
		results = get_clustering_scores(kmeans_predict, samples_test, meta_dict, sample_ids)
		for score_type in results.keys():
			method_col.append("KMeans")
			score_type_col.append(score_type)
			score_col.append(results[score_type])
		# kmedoids clustering
		results = get_score_by_clustering_method("kmedoids", samples_test, meta_dict, sample_ids, args.distance_matrix,
												 args.num_clusters)
		for score_type in results.keys():
			method_col.append("KMedoids")
			score_type_col.append(score_type)
			score_col.append(results[score_type])
		#L2UniFrac
		rep_sample_dict = get_rep_sample_dict(meta_sample_dict, sample_vector_dict, Tint, lint, nodes_in_order)
		results = get_L2UniFrac_accuracy_results(samples_test, targets_test, Tint, lint, nodes_in_order, rep_sample_dict, sample_vector_dict)
		for score_type in results.keys():
			method_col.append("L2UniFrac")
			score_type_col.append(score_type)
			score_col.append(results[score_type])
	df["Method"] = method_col
	df["Score_type"] = score_type_col
	df["Score"] = score_col
	print(df)
	df.to_csv(args.save, sep="\t")


if __name__ == '__main__':
	main()
