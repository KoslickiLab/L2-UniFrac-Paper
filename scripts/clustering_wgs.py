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

#Compare clustering among KMedoids, Kmeans and L2UniFrac
def get_metadata_dict(meta_file, val_col = "HMgDB_diagnosis", key_col = "library_id"):
	meta_dict = dict()
	df = pd.read_csv(meta_file)
	for i, id in enumerate(df[key_col]):
		meta_dict[id] = df[val_col][i]
	return meta_dict

def get_wgs_pushed_vectors(sample_dict, Tint, lint, nodes_in_order):
	'''

	:param sample_dict: assume uniform length.
	:return:
	'''
	pushed_dict = dict()
	for sample, vector in sample_dict.items():
		pushed_dict[sample] = L2U.push_up(vector, Tint, lint, nodes_in_order)
	return pushed_dict

def get_KMedoids_clustering_score(dmatrix_file, n_clusters, meta_dict):
	distance_matrix = pd.read_csv(dmatrix_file, header=None)
	sample_ids = distance_matrix.columns
	labels = get_true_label(meta_dict, sample_ids)
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.')
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?', default="HMgDB_diagnosis")
	parser.add_argument('-dm', '--distance_matrix', type=str, help="Pairwise unifrac distance matrix.")
	parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
	parser.add_argument('-d', '--pdir', type=str, help="Directory of profiles")

	args = parser.parse_args()
	metadata_file = args.meta_file
	metadata_key = args.phenotype
	profile_dir = args.pdir
	distance_matrix = args.distance_matrix

	profile_path_lst = [os.path.join(profile_dir, file) for file in os.listdir(profile_dir)]
	Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_lst)
	meta_dict = get_metadata_dict(metadata_file, val_col=metadata_key)
	n_clusters = len(set(meta_dict.values()))

	all_samples = sample_id = list(meta_dict.keys())
	all_samples_paths = [profile_dir + '/' + sample + '.profile' for sample in all_samples]
	sample_vector_dict = L2U.merge_profiles_by_dir(all_samples_paths, nodes_to_index)

	L2_vectors = push_up_all(sample_vector_dict, Tint, lint, nodes_in_order)
	km_score = get_KMedoids_clustering_score(distance_matrix, n_clusters, meta_dict)

	l2_score = get_L2_clustering_score(L2_vectors, n_clusters, meta_dict)
	print('KMedoids clustering score: ', km_score)
	print('L2UniFrac clustering score: ', l2_score)
