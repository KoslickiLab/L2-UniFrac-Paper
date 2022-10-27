import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
import L2UniFrac as L2U
import pandas as pd
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix #to install: pip install scikit-bio
from sklearn.model_selection import train_test_split
from collections import Counter


#classification
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

def decipher_label_by_vote(predictions, index_sample_dict, group_name, meta_dict):
	'''
	Predict sample for traditional clustering-based method.
	Given a list of predictions and a group name, return predicted class of the group by majority vote
	:param predictions: a list of predicted classes obtained by clustering. e.g.[0,1,0,0,1...0]
	:param index_sample_dict: maps each position of the prediction to its sample name
	:param group_name: An integer. Predicted group name produced from clustering results. e.g. 0, 1
	:param meta_dict: maps a sample to its phenotype. Can be produced from 'get_metadata_dict' from helper
	:return: label predicted by majority vote. e.g. 'skin', 'gut'
	'''
	indices_this_group = [i for i in range(len(predictions)) if predictions[i] == group_name] #indices of predictions that equals to group_name given
	predicted_labels = [meta_dict[index_sample_dict[i]] for i in indices_this_group] #real labels of these samples
	c = Counter(predicted_labels)
	predicted_by_vote = c.most_common(1)[0][0]
	return predicted_by_vote

def get_label_by_proximity(test_sample, rep_sample_dict, Tint, lint, nodes_in_order):
	'''
	Predict sample in L2-UniFrac classification method.
	Given a test_sample and a rep_sample_dict, compute the L2-UniFrac between test_sample and each of the rep_samples in
	rep_sample_dict, and assign the phenotype of which the rep_sample produces the least L2-UniFrac distance to the
	test_sample
	:param test_sample: a vector representation of a sample
	:param rep_sample_dict: {phenotye: vector}
	:param Tint: A dict showing node information of the tree
	:param lint: A dict showing branch information of the tree
	:param nodes_in_order: a list showing the ordering of the nodes on the tree from leaf up
	:return: One of the phenotypes in the keys of rep_sample_dict
	'''
	min_unifrac = 1000
	label = ""
	for phenotype in rep_sample_dict:
		L2unifrac = L2U.L2UniFrac_weighted_plain(Tint, lint, nodes_in_order, test_sample, rep_sample_dict[phenotype])
		if L2unifrac < min_unifrac:
			min_unifrac = L2unifrac
			label = phenotype
	return label

def get_pcoa(dist_matrix, sample_lst, meta_file, col_name, plot_title):
	'''
	Get a PCOA plot based on the distance matrix, colored according to metadata
	:param dist_matrix:
	:param sample_lst:
	:param meta_dict: a dict in the form of {id: {'environment': phenotype}}, can be obtained by calling make_metadata_dict_for_pcoa
	:param plot_title:
	:return:
	'''
	meta_df = pd.read_table(meta_file, sep='\t')
	dm = DistanceMatrix(dist_matrix, sample_lst)
	filtered_meta_df = meta_df[meta_df['sample_name'].isin(sample_lst)]
	filtered_meta_df.set_index('sample_name', inplace=True)
	filtered_meta_df.fillna('unknown', inplace=True)
	dist_pc = pcoa(dm)
	fig = dist_pc.plot(df=filtered_meta_df, column=col_name, cmap="Set1", title=plot_title, axis_labels=('PC1', 'PC2', 'PC3'))
	return fig

def get_metadata_dict(meta_file, val_col = "HMgDB_diagnosis", key_col = "library_id"):
	'''
	Given a file containing metadata, return a dictionary with keys specified in key_col and values specified in val_col
	Usually the key is the id and the value is the phenotype or class a sample belongs to
	:param meta_file: Path to the file containing metadata.
	:param val_col: Column title of the meta_file to be used as keys
	:param key_col: Column title of the meta_file to be used as values
	:return: A dictionary with keys being values of key_col and values being values of val_col. e.g. sample_id:phenotype
	'''
	simple_meta_dict = dict()
	df = pd.read_csv(meta_file, sep='\t')
	for i, id in enumerate(df[key_col]):
		simple_meta_dict[id] = df[val_col][i]
	return simple_meta_dict

def get_meta_samples_dict(simple_meta_dict):
	'''
	Reverse the simple meta_dict ({sample_id: phenotype}) and obtain {phenotype: list of samples}
	:param simple_meta_dict: A {sample_id: phenotype} dictionary, obtained by running get_metadata_dict
	:return: a {phenotype: list of samples} dictionary
	'''
	meta_samples_dict = dict()
	for sample,phenotype in simple_meta_dict.items():
		if phenotype not in meta_samples_dict:
			meta_samples_dict[phenotype] = [sample]
		else:
			meta_samples_dict[phenotype].append(sample)
	return meta_samples_dict

def get_pheno_sample_dict(sample_paths, targets):
	'''
	Given a list of samples and corresponding targets, group them into a dict of {phenotype: [list of sample paths]}
	:param sample_paths: a list of paths leading to sample profiles
	:param targets: a list of targets (phenotypes)
	:return: {phenotype:[list of samples]}
	'''
	pheno_sample_dict = dict()
	for i, pheno in enumerate(targets):
		if pheno in pheno_sample_dict:
			pheno_sample_dict[pheno].append(sample_paths[i])
		else:
			pheno_sample_dict[pheno] = [sample_paths[i]]
	return pheno_sample_dict

def get_rep_sample_dict(pheno_sample_dict, Tint, lint, nodes_in_order, nodes_to_index):
	'''
	Compute the representative sample using the L2 UniFrac method for each phenotype in pheno_sample_dict,
	and return it in a dict. WGS version.
	:param pheno_sample_dict: {phenotype : [list of samples]}
	:param Tint: A dict showing nodes and their respective ancestor
	:param lint: A dict showing edge length
	:param nodes_in_order: Nodes of a tree in order, labeled as integers
	:param nodes_to_index: A dict that maps node name to the labeling in nodes_in_order
	:return: {phenotype:rep_sample}
	'''
	rep_sample_dict = dict()
	for pheno in pheno_sample_dict.keys():
		profile_path_list = pheno_sample_dict[pheno]
		rep_sample = L2U.get_representative_sample_wgs(profile_path_list, Tint, lint, nodes_in_order, nodes_to_index)
		rep_sample_dict[pheno] = rep_sample
	return rep_sample_dict

def write_vector_to_file(vector, file_name, nodes_in_order, nodes_to_index):
	df = pd.DataFrame(columns=['ID', 'relative_abundance'])
	index_to_nodes = {y:x for x, y in nodes_to_index.items()}
	df['ID'] = [index_to_nodes[i] for i in nodes_in_order]
	df['relative_abundance'] = vector
	df.to_csv(file_name, sep='\t', header=True, index=None)
	return

