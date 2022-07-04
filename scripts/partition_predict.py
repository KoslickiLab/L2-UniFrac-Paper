import sys, biom
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
from extract_data import extract_biom, extract_samples, extract_metadata, parse_tree_file, parse_envs
from random import shuffle
from math import floor
import argparse

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

def extract_samples_by_group(biom_file, metadata_file, metadata_key):
	'''

	:param biom_file: A .biom file
	:param metadata_file: Metadata file
	:param metadata_key: The phenotype of interest. For example, 'body sites'
	:return:
	'''
	Biom = biom.load_table(biom_file)
	sample_ids = Biom.ids()
	tree_nodes = Biom.ids(axis='observation')
	metadata = extract_metadata(metadata_file)
	nodes_test = extract_biom(biom_file)
	group_samples = {}
	for sample in sample_ids:
		col = []
		for node in tree_nodes:
			col.append(nodes_test[node][sample])
		if metadata[sample][metadata_key] not in group_samples:
			group_samples[metadata[sample][metadata_key]] = []
		group_samples[metadata[sample][metadata_key]].append(col)

	return group_samples

def extract_sample_names_by_group(biom_file, metadata_file, metadata_key):
	Biom = biom.load_table(biom_file)
	sample_ids = Biom.ids()
	metadata = extract_metadata(metadata_file)
	group_name_samples = {}
	for sample in sample_ids:
		if metadata[sample][metadata_key] not in group_name_samples:
			group_name_samples[metadata[sample][metadata_key]] = []
		group_name_samples[metadata[sample][metadata_key]].append(sample)

	return group_name_samples

def extract_samples_direct(biom_file, tree_file):
	nodes_samples = extract_biom(biom_file)
	_, _, nodes_in_order = parse_tree_file(tree_file)
	(nodes_weighted, samples_temp) = parse_envs(nodes_samples, nodes_in_order)
	sample_ids = extract_samples(biom_file)

	return nodes_weighted, sample_ids

def extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key):
	nodes_samples = extract_biom(biom_file)
	_, _, nodes_in_order = parse_tree_file(tree_file)
	(nodes_weighted, samples_temp) = parse_envs(nodes_samples, nodes_in_order)
	sample_ids = extract_samples(biom_file)

	metadata = extract_metadata(metadata_file)
	group_name_samples = {}
	for sample in sample_ids:
		if metadata[sample][metadata_key] not in group_name_samples:
			group_name_samples[metadata[sample][metadata_key]] = {}
		group_name_samples[metadata[sample][metadata_key]][sample] = nodes_weighted[sample]

	return group_name_samples, sample_ids, list(group_name_samples.keys())

# Partition between train and test randomly
def partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key):
	'''

	:param train_percentage: what percentage of the sample should be training samples.
	:param biom_file: a .biom file
	:param tree_file:
	:param metadata_file:
	:param metadata_key: The phenotype of interest. For example, 'body sites'
	:return:
	'''
	try:
		assert train_percentage <= 90 and train_percentage >= 10
	except:
		raise TrainingRateTooHighOrLow(train_percentage)

	group_name_samples, sample_ids, classes = extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key)
	train_dict = {}
	test_dict = {}
	for c in classes:
		if c not in train_dict:
			train_dict[c] = {}
		if c not in test_dict:
			test_dict[c] = {}
		class_samples = [(key, value) for key, value in group_name_samples[c].items()]
		l = len(class_samples)
		try:
			assert l > 100
		except:
			raise ClassTooSmall(c)
		train_num = floor(l*(train_percentage/100))
		test_num = l - train_num
		base_list = [0 for i in range(train_num)] + [1 for i in range(test_num)]
		shuffled_list = shuffle(base_list)
		for i in range(len(base_list)):
			if base_list[i] == 0:
				train_dict[c][class_samples[i][0]] = class_samples[i][1]
			if base_list[i] == 1:
				test_dict[c][class_samples[i][0]] = class_samples[i][1]

	return train_dict, test_dict

biom_file = 'data/biom/47422_otu_table.biom'
metadata_file = 'data/metadata/P_1928_65684500_raw_meta.txt'
tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
metadata_key = 'body_site'
train_percentage = 80
#extract_samples_by_group(biom_file, metadata_file, metadata_key)
#extract_sample_names_by_group(biom_file, metadata_file, metadata_key)
#extract_samples_direct(biom_file, tree_file)
#extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key)
#partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-m', '--meta_file', type=str, help='A metadata file.')
	parser.add_argument('-d', '--dir', type=str, help='A directory containing profiles. Only required if data type is wgs.')
	parser.add_argument('-msg', '--message', type=str, help='Message printed in the output file before test statistics.')
	parser.add_argument('-p', '--phenotype', type=str, help='A selected phenotype corresponding to a column name in the metadata file.')
	parser.add_argument('-t', '--data_type', type=str, help='wgs or 16s.')
	parser.add_argument('-ot', '--otu_table', type=str, help='Path to the otu table.')

	args = parser.parse_args()
	#if args.data_type == '16s':
	train_dict, test_dict = partition_samples(train_percentage, biom_file, tree_file, metadata_file, metadata_key)
	print(len(train_dict.keys()))
	print(len(test_dict.keys()))
	print(test_dict)
