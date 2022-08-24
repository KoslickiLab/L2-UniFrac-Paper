import sys, os, argparse, torch, csv, dendropy
sys.path.append('../L2-UniFrac')
sys.path.append('../L2-UniFrac/src')
sys.path.append('../L2-UniFrac/scripts')
from torch import nn, optim, FloatTensor, LongTensor, cuda, tensor, no_grad
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from random import shuffle
from math import floor, log
from statistics import mean
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import rand_score, accuracy_score, recall_score, precision_score, f1_score
from extract_data import extract_biom_samples, extract_samples, extract_metadata_direct, extract_sample_metadata
from sklearn.model_selection import train_test_split
import L2UniFrac as L2U

class ResNet(nn.Module):
	def __init__(self, l1_in, l2_size, l3_out):
		super().__init__()
		self.l1 = nn.Linear(l1_in, l2_size)
		self.l2 = nn.Linear(l2_size, l2_size)
		self.l3 = nn.Linear(l2_size, l3_out)
		self.do = nn.Dropout(0.1)

	def forward(self, x):
		h1 = nn.functional.relu(self.l1(x))
		h2 = nn.functional.relu(self.l2(h1))
		do = self.do(h2 + h1)
		logits = self.l3(do)
		return logits

def get_metadata_dict(meta_file, val_col = "HMgDB_diagnosis", key_col = "library_id"):
	meta_dict = dict()
	df = pd.read_csv(meta_file)
	for i, id in enumerate(df[key_col]):
		meta_dict[id] = df[val_col][i]
	return meta_dict

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

def get_pheno_sample_dict(samples_path_train, targets_train):
	pheno_sample_dict = dict()
	for i, pheno in enumerate(targets_train):
		if pheno in pheno_sample_dict:
			pheno_sample_dict[pheno].append(samples_path_train[i])
		else:
			pheno_sample_dict[pheno] = [samples_path_train[i]]
	return pheno_sample_dict

def prepare_data_16s(biom_file, metadata_file):
	nodes_samples = extract_biom_samples(biom_file)
	sample_ids = extract_samples(biom_file)
	metadata = extract_metadata_direct(metadata_file)
	sample_metadata = extract_sample_metadata(biom_file, metadata_file)

	return nodes_samples, sample_ids, metadata, sample_metadata

def prepare_inputs_16s(biom_file, metadata_file, batch_size, test_size):
	nodes_samples, sample_ids, metadata, sample_metadata = prepare_data_16s(biom_file, metadata_file)

	l = len(sample_ids)
	train_num = floor(l*(1-test_size))
	test_num = l - train_num
	base_list = [0 for i in range(train_num)] + [1 for i in range(test_num)]
	shuffled_list = shuffle(base_list)
	shuffled_list = base_list
	
	train_samples = []
	test_samples = []
	for i, sample in enumerate(sample_ids):
		if shuffled_list[i] == 0:
			train_samples.append(sample)
		else:
			test_samples.append(sample)

	train_loader = []
	test_loader = []
	tmp_x = []
	tmp_y = []
	curr_len = 0
	for sample in train_samples:
		sample_p = nodes_samples[sample]
		meta_p = sample_metadata[sample]
		tmp_x.append(sample_p)
		tmp_y.append(meta_p)
		curr_len += 1
		if curr_len == batch_size:
			train_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])
			curr_len = 0
			tmp_x = []
			tmp_y = []

	if len(tmp_x) > 0:
		train_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])

	tmp_x = []
	tmp_y = []
	curr_len = 0
	for sample in test_samples:
		sample_p = nodes_samples[sample]
		meta_p = sample_metadata[sample]
		tmp_x.append(sample_p)
		tmp_y.append(meta_p)
		curr_len += 1
		if curr_len == batch_size:
			test_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])
			curr_len = 0
			tmp_x = []
			tmp_y = []

	if len(tmp_x) > 0:
		test_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])

	return train_loader, test_loader

def prepare_data_wgs(profile_dir, metadata_file, phenotype, test_size):
	profile_path_lst = [os.path.join(profile_dir, file) for file in os.listdir(profile_dir)]
	Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_lst)
	meta_dict = get_metadata_dict(metadata_file, val_col=phenotype)
	train_samples, test_samples, targets_train, targets_test = partition_sample(meta_dict, random_state=0, test_size=test_size)
	samples_train_paths = [profile_dir + '/' + sample + '.profile' for sample in train_samples]
	#pheno_sample_dict = get_pheno_sample_dict(samples_train_paths, targets_train)
	train_sample_dict = L2U.merge_profiles_by_dir(samples_train_paths, nodes_to_index)
	samples_test_paths = [profile_dir + '/' + sample + '.profile' for sample in test_samples]
	test_sample_dict = L2U.merge_profiles_by_dir(samples_test_paths, nodes_to_index)

	return meta_dict, train_samples, test_samples, train_sample_dict, test_sample_dict

def prepare_inputs_wgs(profile_dir, metadata_file, phenotype, batch_size, include_adenoma, test_size):
	meta_dict, train_samples, test_samples, train_sample_dict, test_sample_dict = prepare_data_wgs(profile_dir, metadata_file, phenotype, test_size)

	classifications = list(set(list(meta_dict.values())))
	class_dict = {classifications[i]:i for i in range(len(classifications))}

	for k, v in meta_dict.items():
		meta_dict[k] = class_dict[v]

	train_loader = []
	test_loader = []
	tmp_x = []
	tmp_y = []
	curr_len = 0
	for sample in train_samples:
		sample_p = train_sample_dict[sample]
		meta_p = meta_dict[sample]
		if (not include_adenoma and meta_p != class_dict['adenoma']) or include_adenoma:
			tmp_x.append(sample_p)
			tmp_y.append(meta_p)
			curr_len += 1
			if curr_len == batch_size:
				train_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])
				curr_len = 0
				tmp_x = []
				tmp_y = []

	if len(tmp_x) > 0:
		train_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])

	tmp_x = []
	tmp_y = []
	curr_len = 0
	for sample in test_samples:
		sample_p = test_sample_dict[sample]
		meta_p = meta_dict[sample]
		if (not include_adenoma and meta_p != class_dict['adenoma']) or include_adenoma:
			tmp_x.append(sample_p)
			tmp_y.append(meta_p)
			curr_len += 1
			if curr_len == batch_size:
				test_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])
				curr_len = 0
				tmp_x = []
				tmp_y = []

	if len(tmp_x) > 0:
		test_loader.append([FloatTensor([tmp_x]), LongTensor(tmp_y)])

	return train_loader, test_loader

def train_model(model, train_loader, test_loader, nb_epochs, verbose=False):
	# Begin Training
	for epoch in range(nb_epochs):
		losses = list()
		accuracies = list()
		model.train()
		for batch in train_loader:
			x, y = batch
			b = x.size(0)
			if cuda.is_available():
				x = x[0].cuda()
			else:
				x = x[0]

			# 1) Forward
			l = model(x) # logits

			# 2) Compute the Objective Function
			if cuda.is_available():
				J = loss(l, y.cuda())
			else:
				J = loss(l, y)

			# 3) Clean the Gradients
			model.zero_grad()

			# 4) Accumulate the Partial Derivatives of J w.r.t. Parameters
			J.backward()

			# 5) Step in the Opposite Direction of the Gradient
			optimizer.step()

			losses.append(J.item())
			if cuda.is_available():
				accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
			else:
				accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

		if verbose:
			print(f'Epoch {epoch + 1}, training loss: {tensor(losses).mean():.2f}, training accuracy: {tensor(accuracies).mean():.2f}')

		if verbose:
			losses = list()
			accuracies = list()
			model.eval()
			for batch in test_loader:
				x, y = batch

				b = x.size(0)
				if cuda.is_available():
					x = x[0].cuda()
				else:
					x = x[0]

				# 1) Forward
				with no_grad():
					l = model(x) # logits

				# 2) Compute the Objective Function
				if cuda.is_available():
					J = loss(l, y.cuda())
				else:
					J = loss(l, y)

				losses.append(J.item())

				if cuda.is_available():
					accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
				else:
					accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

			print(f'Epoch {epoch + 1}, validation loss: {tensor(losses).mean():.2f}, validation accuracy: {tensor(accuracies).mean():.2f}')

	return model

def test_model(model, test_loader):
	# Classify using model:
	classes_test = []
	losses = list()
	accuracies = list()
	model.eval()
	for batch in test_loader:
		x, y = batch

		b = x.size(0)
		if cuda.is_available():
			x = x[0].cuda()
		else:
			x = x[0]

		# 1) Forward
		with no_grad():
			l = model(x) # logits

		# 2) Compute the Objective Function
		if cuda.is_available():
			J = loss(l, y.cuda())
		else:
			J = loss(l, y)

		losses.append(J.item())

		if cuda.is_available():
			accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
		else:
			accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

		classes_test = classes_test + l.detach().argmax(dim=1).tolist()

	classes_real = []
	for x, y in test_loader:
		classes_real = classes_real + y.tolist()

	return classes_real, classes_test

def print_results(indents, RI, ARI, NMI, AMI, FM, AC, RE, PR, F1):
	tabs = '\t'*indents
	print(f'{tabs}Rand Index Score:               {RI}')
	print(f'{tabs}Adjusted Rand Index Score:      {ARI}')
	print(f'{tabs}Normalized Mutual Index Score:  {NMI}')
	print(f'{tabs}Adjusted Mutual Info Score:     {AMI}')
	print(f'{tabs}Fowlkes Mallows Score:          {FM}')
	print(f'{tabs}Accuracy Score:          \t\t{AC}')
	print(f'{tabs}Recall Score:        	 \t\t{RE}')
	print(f'{tabs}Precision Score:         \t\t{PR}')
	print(f'{tabs}F1 Score:        		 \t\t{F1}')

def run_scoring(classes_real, classes_test, verbose=True):
	RI = rand_score(classes_real, classes_test)
	ARI = adjusted_rand_score(classes_real, classes_test)
	NMI = normalized_mutual_info_score(classes_real, classes_test)
	AMI = adjusted_mutual_info_score(classes_real, classes_test)
	FM = fowlkes_mallows_score(classes_real, classes_test)
	AC = accuracy_score(classes_real, classes_test)
	RE = recall_score(classes_real, classes_test, average='macro', zero_division=0)
	PR = precision_score(classes_real, classes_test, average='macro', zero_division=0)
	F1 = f1_score(classes_real, classes_test, average='macro', zero_division=0)

	print_results(2, RI, ARI, NMI, AMI, FM, AC, RE, PR, F1)

	return RI, ARI, NMI, AMI, FM, AC, RE, PR, F1

if __name__ == '__main__':
	
	test_sizes = [0.3, 0.2, 0.1]
	run_n = 10

	parser = argparse.ArgumentParser(description='Get testing statistics of classification test.')
	parser.add_argument('-d', '--data_type', type=str, help='16s data or WGS data', nargs='?', default='wgs')
	parser.add_argument('-b', '--biom_file', type=str, help='Path to biom file', nargs='?', default='../data/biom/47422_otu_table.biom')
	parser.add_argument('-m', '--metadata_file', type=str, help='Path to metadata file', nargs='?', default='../data/hmgdb_adenoma_bioproject266076.csv')
	parser.add_argument('-p', '--profile_dir', type=str, help='Path to profiles', nargs='?', default='../data/adenoma_266076/profiles')
	parser.add_argument('-ph', '--phenotype', type=str, help='Phenotype to use on WGS', nargs='?', default='HMgDB_diagnosis')
	parser.add_argument('-mi', '--model_in', type=int, help='Layer 1 input dimension', nargs='?', default=1749)
	parser.add_argument('-mo', '--model_out', type=int, help='Layer 3 output dimension', nargs='?', default=3)
	parser.add_argument('-ba', '--batch_size', type=int, help='Batch size to use', nargs='?', default=8)
	parser.add_argument('-a', '--include_adenoma', type=bool, help='Include or exclude adenoma in WGS', nargs='?', default=True)
	parser.add_argument('-e', '--nb_epochs', type=int, help='Number of training epochs', nargs='?', default=50)

	args = parser.parse_args()
	useData = args.data_type #'wgs'
	biom_file_16s = args.biom_file #'../data/biom/47422_otu_table.biom'
	metadata_file_16s = args.metadata_file #'../data/metadata/P_1928_65684500_raw_meta.txt'
	profile_dir_wgs = args.profile_dir #'../data/adenoma_266076/profiles'
	metadata_file_wgs = args.metadata_file #'../data/hmgdb_adenoma_bioproject266076.csv'
	phenotype_wgs = args.phenotype #'HMgDB_diagnosis'
	model_in_16s = args.model_in #9160
	model_out_16s = args.model_out #5
	model_in_wgs = args.model_in #1749
	model_out_wgs = args.model_out #3
	batch_size_16s = args.batch_size #32
	batch_size_wgs = args.batch_size #8
	include_adenoma_wgs = args.include_adenoma #True
	nb_epochs = args.nb_epochs #50

	model_intermediate_16s = 2**floor(log(model_in_16s, 2)-2)
	model_intermediate_wgs = 2**floor(log(model_in_wgs, 2)-2)

	for test_size in test_sizes:
		print(f'Running on {int(test_size*100)}% Testing Size:')
		RI_list, ARI_list, NMI_list, AMI_list, FM_list, AC_list, RE_list, PR_list, F1_list = [], [], [], [], [], [], [], [], []
		for run in range(run_n):
			print(f'\tRun {run+1} Results:')
			if useData == '16s':
				if cuda.is_available():
					model = ResNet(model_in_16s, model_intermediate_16s, model_out_16s).cuda()
				else:
					model = ResNet(model_in_16s, model_intermediate_16s, model_out_16s)

				train_loader, test_loader = prepare_inputs_16s(biom_file_16s, metadata_file_16s, batch_size_16s, test_size)
			elif useData == 'wgs':
				if cuda.is_available():
					model = ResNet(model_in_wgs, model_intermediate_wgs, model_out_wgs).cuda()
				else:
					model = ResNet(model_in_wgs, model_intermediate_wgs, model_out_wgs)

				train_loader, test_loader = prepare_inputs_wgs(profile_dir_wgs, metadata_file_wgs, phenotype_wgs, batch_size_wgs, include_adenoma_wgs, test_size)

			optimizer = optim.SGD(model.parameters(), lr=1e-2)

			loss = nn.CrossEntropyLoss()

			model = train_model(model, train_loader, test_loader, nb_epochs)	

			classes_real, classes_test = test_model(model, test_loader)

			RI, ARI, NMI, AMI, FM, AC, RE, PR, F1 = run_scoring(classes_real, classes_test)
			RI_list.append(RI)
			ARI_list.append(ARI)
			NMI_list.append(NMI)
			AMI_list.append(AMI)
			FM_list.append(FM)
			AC_list.append(AC)
			RE_list.append(RE)
			PR_list.append(PR)
			F1_list.append(F1)

		print(f'\tAverage results of {int(test_size*100)}% Testing Size:')
		RI = mean(RI_list)
		ARI = mean(ARI_list)
		NMI = mean(NMI_list)
		AMI = mean(AMI_list)
		FM = mean(FM_list)
		AC = mean(AC_list)
		RE = mean(RE_list)
		PR = mean(PR_list)
		F1 = mean(F1_list)
		print_results(2, RI, ARI, NMI, AMI, FM, AC, RE, PR, F1)