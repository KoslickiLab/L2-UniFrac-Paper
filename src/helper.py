import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
import L2UniFrac as L2U
import pandas as pd

def get_metadata_dict(meta_file, val_col = "HMgDB_diagnosis", key_col = "library_id"):
	'''
	Parse a metadata file to obtain a dict with keys in key_col and values in val_col
	:param meta_file: a metadata file containing columns that can be used as keys and values
	:param val_col: column name for values
	:param key_col: column name for keys
	:return: a dict with keys in key_col and values in val_col
	'''
	meta_dict = dict()
	df = pd.read_csv(meta_file)
	for i, id in enumerate(df[key_col]):
		meta_dict[id] = df[val_col][i]
	return meta_dict

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
	and return it in a dict.
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

def write_vector_to_file(vector, save_dir, save_name, nodes_in_order, nodes_to_index):
	df = pd.DataFrame(columns=['ID', 'relative_abundance'])
	index_to_nodes = {y:x for x, y in nodes_to_index.items()}
	df['ID'] = [index_to_nodes[i] for i in nodes_in_order]
	file_name = save_dir + '/' + save_name + '.txt'
	df['relative_abundance'] = vector
	df.to_csv(file_name, sep='\t', header=False)
	return

