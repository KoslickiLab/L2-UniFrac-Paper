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

def get_pcoa(dist_matrix, sample_lst, meta_file, col_name, plot_title):
	'''
	Get a PCOA plot based on the distance matrix, colored according to metadata
	:param dist_matrix:
	:param sample_lst:
	:param meta_dict: a dict in the form of {id: {'environment': phenotype}}, can be obtained by calling make_metadata_dict_for_pcoa
	:param plot_title:
	:return:
	'''
	df = pd.read_table(meta_file)
	print(df.head())
	dm = DistanceMatrix(dist_matrix, sample_lst)
	dist_pc = pcoa(dm)
	return dist_pc.plot(df=df, column=col_name, cmap="Set1", title=plot_title, axis_labels=('PC1', 'PC2', 'PC3'))

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
	df = pd.read_csv(meta_file)
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

