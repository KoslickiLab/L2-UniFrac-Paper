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
from extract_data import extract_biom, extract_samples, extract_sample_metadata, parse_tree_file, parse_envs, extract_biom_samples

#Compare clustering among KMedoids, Kmeans and L2UniFrac

def get_wgs_pushed_vectors(sample_dict, Tint, lint, nodes_in_order):
	'''

	:param sample_dict: assume uniform length.
	:return:
	'''
	pushed_dict = dict()
	for sample, vector in sample_dict.items():
		pushed_dict[sample] = L2U.push_up(vector, Tint, lint, nodes_in_order)
	return pushed_dict

def get_KMedoids_clustering_score(dmatrix_file, n_clusters, labels):
	distance_matrix = pd.read_csv(dmatrix_file, header=0, index_col=0, sep='\t')
	kmedoids_prediction = KMedoids(n_clusters=n_clusters, metric='precomputed', method='pam', init='heuristic').fit_predict(distance_matrix)
	return fowlkes_mallows_score(kmedoids_prediction, labels)

def get_L2_clustering_score(L2_samples, n_clusters, labels):
	all_vectors = list(L2_samples)
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
	print(len(sample_ids))
	meta_dict = extract_sample_metadata(biom_file, metadata_file)
	n_clusters = args.num_clusters

	sample_vector = extract_biom_samples(biom_file)
	L2_vectors = push_up_all(sample_vector, Tint, lint, nodes_in_order)

	labels = get_true_label(meta_dict, sample_ids)
	km_score = get_KMedoids_clustering_score(distance_matrix, n_clusters, labels)
	l2_score = get_L2_clustering_score(L2_vectors, n_clusters, labels)
