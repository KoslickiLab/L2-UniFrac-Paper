import sys, biom, dendropy
sys.path.append('../L2-Unifrac')
sys.path.append('../L2-Unifrac/src')
sys.path.append('../L2-Unifrac/scripts')
from data import extract_biom, extract_samples, extract_metadata, parse_tree_file, parse_envs
from random import shuffle
from math import floor

class TrainingRateTooHighOrLow(Exception):

    def __init__(self, train_rate, msg="Invalid training rate. Rate should be between 10 and 90."):
        self.train_rate = train_rate
        self.msg = msg
        super().__init__(self.msg)

class ClassTooSmall(Exception):

    def __init__(self, c, msg="Class size too small. Minimum size is 100 samples."):
        self.c = c
        self.msg = msg
        super().__init__(self.msg)

def extract_samples_by_group(biom_file, metadata_file, metadata_key):
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
def partition_samples(train_rate, biom_file, tree_file, metadata_file, metadata_key):
	
	try:
		assert train_rate <= 90 and train_rate >= 10
	except:
		raise TrainingRateTooHighOrLow(train_rate)

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
		train_num = floor(l*(train_rate/100))
		test_num = l - train_num
		base_list = [0 for i in range(train_num)] + [1 for i in range(test_num)]
		shuffled_list = shuffle(base_list)
		for i in range(len(base_list)):
			if base_list[i] == 0:
				train_dict[c][class_samples[i][0]] = class_samples[i][1]
			if base_list[i] == 1:
				test_dict[c][class_samples[i][0]] = class_samples[i][1]

	return train_dict, test_dict

biom_file = '../data/biom/47422_otu_table.biom'
metadata_file = '../data/metadata/P_1928_65684500_raw_meta.txt'
tree_file = '../data/trees/gg_13_5_otus_99_annotated.tree'
metadata_key = 'body_site'
train_rate = 80
#extract_samples_by_group(biom_file, metadata_file, metadata_key)
#extract_sample_names_by_group(biom_file, metadata_file, metadata_key)
#extract_samples_direct(biom_file, tree_file)
#extract_samples_direct_by_group(biom_file, tree_file, metadata_file, metadata_key)
partition_samples(train_rate, biom_file, tree_file, metadata_file, metadata_key)