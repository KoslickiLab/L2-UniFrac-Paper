import sys
import os
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
sys.path.append('src')
import argparse
import itertools as it
import L2UniFrac as L2U
from helper import get_metadata_dict, get_profile_path_list, get_profile_name_list,\
	get_pheno_sample_dict, get_rep_sample_dict_wgs_component_wise_mean, get_taxonomy_in_order

def parse_arguments():
	parser = argparse.ArgumentParser(description='Get pairwise diff abund plots.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.', nargs='?')
	parser.add_argument('-id_col', '--id_col', type=str, help='Name of the id column in the metadata file.', nargs='?',
						default="library_id")
	parser.add_argument('-d', '--pdir', type=str, help="Directory of profiles")
	parser.add_argument('-p', '--phenotype', type=str,
						help='A selected phenotype corresponding to a column name in the metadata file.', nargs='?',
						default="HMgDB_diagnosis")
	parser.add_argument('-s', '--save_dir', type=str, help='Directory to save images under.')
	parser.add_argument('-prefix', '--prefix', type=str, help='Prefix for the saved files.')
	parser.add_argument('-t', '--tax_rank', type=str, help='Include only this taxon level.', choices=['suerkingdom', 'phylum',
																									  'class', 'order',
																									  'family', 'genus',
																									  'species'],
						nargs='?', default='species')
	return parser.parse_args()


def get_diff_abund_plot(Tint, lint, nodes_in_order, nodes_to_index, P, Q, P_label, Q_label, max_tax_rank):
	(Z, diffabund) = L2U.L2UniFrac_weighted(Tint, lint, nodes_in_order, P, Q)
	taxid_in_order = get_taxonomy_in_order(nodes_in_order, nodes_to_index)
	fig = L2U.plot_diffab_by_tax(nodes_in_order, taxid_in_order, diffabund, P_label, Q_label, max_tax_rank = max_tax_rank,
								 thresh=0.0005, includeTemp=False, show=False)
	return fig


def main(my_args):
	profile_path_list = get_profile_path_list(my_args.pdir)
	profile_name_list = get_profile_name_list(my_args.pdir)
	if my_args.save_dir is None:
		my_args.save_dir = my_args.pdir
	meta_dict = get_metadata_dict(my_args.meta_file)
	Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_list)
	targets = [meta_dict[i] for i in profile_name_list]
	pheno_sample_dict = get_pheno_sample_dict(profile_path_list, targets)
	rep_sample_dict = get_rep_sample_dict_wgs_component_wise_mean(pheno_sample_dict, nodes_to_index)
	for pair in it.combinations(rep_sample_dict, 2):  #all pairwise combinations of rep vectors
		P_label, Q_label = pair[0], pair[1]
		print(P_label)
		print(Q_label)
		print(my_args.prefix)
		file_name = my_args.prefix + P_label + "_" + Q_label + "_diffabund.png"
		full_file_name = os.path.join(my_args.save_dir, file_name)
		P, Q = rep_sample_dict[P_label], rep_sample_dict[Q_label]
		fig = get_diff_abund_plot(Tint, lint, nodes_in_order, nodes_to_index, P, Q, P_label, Q_label, my_args.tax_rank.lower())
		fig.savefig(full_file_name)

if __name__ == '__main__':
	my_args = parse_arguments()
	main(my_args)

