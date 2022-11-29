import os
import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import argparse
import L2UniFrac as L2U
import numpy as np
import pandas as pd
from helper import get_metadata_dict, get_pheno_sample_dict, get_rep_sample_dict_wgs, write_vector_to_file, get_rep_sample_dict_wgs_component_wise_mean


def generate_rep_sample_from_metadata(meta_dict, profile_dir, save_dir):
	'''

	:param meta_dict:
	:param profile_list:
	:param save_dir:
	:return:
	'''
	profile_list = os.listdir(profile_dir)
	for profile_name in profile_list:
		if not profile_name.endswith('.profile'):
			profile_list.remove(profile_name)
	profile_name_list = list(map(lambda x: x.split('.')[0], profile_list))
	profile_path_lst = [os.path.join(profile_dir, file) for file in profile_list]
	Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_lst)
	targets = [meta_dict[i] for i in profile_name_list]
	#profile list should come from meta_dict
	pheno_sample_dict = get_pheno_sample_dict(profile_path_lst, targets)
	rep_sample_dict = get_rep_sample_dict_wgs_component_wise_mean(pheno_sample_dict, nodes_to_index)
	for pheno in rep_sample_dict.keys():
		file_name = save_dir + '/' + str(pheno) + '_direct.txt'
		write_vector_to_file(rep_sample_dict[pheno], file_name, nodes_in_order, nodes_to_index)
	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get representative samples from a metadata file.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.', nargs='?')
	parser.add_argument('-id_col', '--id_col', type=str, help='Name of the id column in the metadata file.', nargs='?', default="library_id")
	parser.add_argument('-s', '--save', type=str, help="Directory to save files under.")
	parser.add_argument('-d', '--pdir', type=str, help="Directory of profiles")
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?', default="HMgDB_diagnosis")

	args = parser.parse_args()
	metadata_file = args.meta_file
	profile_dir = args.pdir
	metadata_key = args.phenotype
	id_col = args.id_col
	save_dir = args.save

	meta_dict = get_metadata_dict(metadata_file, val_col=metadata_key, key_col=id_col)
	generate_rep_sample_from_metadata(meta_dict, profile_dir, save_dir)

