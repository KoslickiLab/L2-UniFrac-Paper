import sys, os, math
import biom, torch, csv, dendropy
sys.path.append('../L2-UniFrac')
sys.path.append('../L2-UniFrac/src')
sys.path.append('../L2-UniFrac/scripts')
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from random import shuffle
from math import floor
import L2UniFrac as L2U
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import rand_score, accuracy_score
from extract_data import extract_biom_samples, extract_samples, extract_metadata_direct, extract_sample_metadata
from sklearn.model_selection import train_test_split

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

def prepare_inputs_16s(biom_file, metadata_file, batch_size):
	nodes_samples, sample_ids, metadata, sample_metadata = prepare_data_16s(biom_file, metadata_file)

	l = len(sample_ids)
	train_num = floor(l*(80/100))
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
			train_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])
			curr_len = 0
			tmp_x = []
			tmp_y = []

	train_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])

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
			test_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])
			curr_len = 0
			tmp_x = []
			tmp_y = []

	test_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])

	return train_loader, test_loader

def prepare_data_wgs(profile_dir, metadata_file, phenotype):
	profile_path_lst = [os.path.join(profile_dir, file) for file in os.listdir(profile_dir)]
	Tint, lint, nodes_in_order, nodes_to_index = L2U.get_wgs_tree(profile_path_lst)
	meta_dict = get_metadata_dict(metadata_file, val_col=phenotype)
	train_samples, test_samples, targets_train, targets_test = partition_sample(meta_dict, random_state=0, test_size=0.2)
	samples_train_paths = [profile_dir + '/' + sample + '.profile' for sample in train_samples]
	#pheno_sample_dict = get_pheno_sample_dict(samples_train_paths, targets_train)
	train_sample_dict = L2U.merge_profiles_by_dir(samples_train_paths, nodes_to_index)
	samples_test_paths = [profile_dir + '/' + sample + '.profile' for sample in test_samples]
	test_sample_dict = L2U.merge_profiles_by_dir(samples_test_paths, nodes_to_index)

	return meta_dict, train_samples, test_samples, train_sample_dict, test_sample_dict

def prepare_inputs_wgs(profile_dir, metadata_file, phenotype, batch_size, include_adenoma):
	meta_dict, train_samples, test_samples, train_sample_dict, test_sample_dict = prepare_data_wgs(profile_dir, metadata_file, phenotype)

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
		if not include_adenoma and meta_p != class_dict['adenoma']:
			tmp_x.append(sample_p)
			tmp_y.append(meta_p)
			curr_len += 1
			if curr_len == batch_size:
				train_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])
				curr_len = 0
				tmp_x = []
				tmp_y = []

	train_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])

	tmp_x = []
	tmp_y = []
	curr_len = 0
	for sample in test_samples:
		sample_p = test_sample_dict[sample]
		meta_p = meta_dict[sample]
		if not include_adenoma and meta_p != class_dict['adenoma']:
			tmp_x.append(sample_p)
			tmp_y.append(meta_p)
			curr_len += 1
			if curr_len == batch_size:
				test_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])
				curr_len = 0
				tmp_x = []
				tmp_y = []

	test_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])

	return train_loader, test_loader

if __name__ == '__main__':
	useData = 'wgs'
	biom_file_16s = '../data/biom/47422_otu_table.biom'
	metadata_file_16s = '../data/metadata/P_1928_65684500_raw_meta.txt'
	profile_dir_wgs = '../data/adenoma_266076/profiles'
	metadata_file_wgs = '../data/hmgdb_adenoma_bioproject266076.csv'
	phenotype_wgs = 'HMgDB_diagnosis'
	model_in_16s = 9160
	model_out_16s = 5
	model_in_wgs = 1749
	model_out_wgs = 3
	batch_size = 32
	include_adenoma_wgs = True

	model_intermediate_16s = 2**math.floor(math.log(model_in_16s, 2)-2)
	model_intermediate_wgs = 2**math.floor(math.log(model_in_wgs, 2)-2)

	if useData == '16s':
		if torch.cuda.is_available():
			model = ResNet(model_in_16s, model_intermediate_16s, model_out_16s).cuda()
		else:
			model = ResNet(model_in_16s, model_intermediate_16s, model_out_16s)

		train_loader, test_loader = prepare_inputs_16s(biom_file_16s, metadata_file_16s, batch_size)
	elif useData == 'wgs':
		if torch.cuda.is_available():
			model = ResNet(model_in_wgs, model_intermediate_wgs, model_out_wgs).cuda()
		else:
			model = ResNet(model_in_wgs, model_intermediate_wgs, model_out_wgs)

		train_loader, test_loader = prepare_inputs_wgs(profile_dir_wgs, metadata_file_wgs, phenotype_wgs, batch_size, include_adenoma_wgs)

	optimizer = optim.SGD(model.parameters(), lr=1e-2)

	loss = nn.CrossEntropyLoss()

	# Begin Training
	nb_epochs = 50
	classes_test = []
	for epoch in range(nb_epochs):
		losses = list()
		accuracies = list()
		model.train()
		for batch in train_loader:
			x, y = batch
			b = x.size(0)
			if torch.cuda.is_available():
				x = x[0].cuda()
			else:
				x = x[0]

			# 1) Forward
			l = model(x) # logits

			# 2) Compute the Objective Function
			if torch.cuda.is_available():
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
			if torch.cuda.is_available():
				accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
			else:
				accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

		print(f'Epoch {epoch + 1}, training loss: {torch.tensor(losses).mean():.2f}, training accuracy: {torch.tensor(accuracies).mean():.2f}')

		losses = list()
		accuracies = list()
		model.eval()
		for batch in test_loader:
			x, y = batch

			b = x.size(0)
			if torch.cuda.is_available():
				x = x[0].cuda()
			else:
				x = x[0]

			# 1) Forward
			with torch.no_grad():
				l = model(x) # logits

			# 2) Compute the Objective Function
			if torch.cuda.is_available():
				J = loss(l, y.cuda())
			else:
				J = loss(l, y)

			losses.append(J.item())

			if torch.cuda.is_available():
				accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
			else:
				accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

		print(f'Epoch {epoch + 1}, validation loss: {torch.tensor(losses).mean():.2f}, validation accuracy: {torch.tensor(accuracies).mean():.2f}')

	# Classify using model:
	for batch in test_loader:
		x, y = batch

		b = x.size(0)
		if torch.cuda.is_available():
			x = x[0].cuda()
		else:
			x = x[0]

		# 1) Forward
		with torch.no_grad():
			l = model(x) # logits

		# 2) Compute the Objective Function
		if torch.cuda.is_available():
			J = loss(l, y.cuda())
		else:
			J = loss(l, y)

		losses.append(J.item())

		if torch.cuda.is_available():
			accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
		else:
			accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

		classes_test = classes_test + l.detach().argmax(dim=1).tolist()

	classes_real = []
	for x, y in test_loader:
		classes_real = classes_real + y.tolist()

	RI = rand_score(classes_real, classes_test)
	ARI = adjusted_rand_score(classes_real, classes_test)
	NMI = normalized_mutual_info_score(classes_real, classes_test)
	AMI = adjusted_mutual_info_score(classes_real, classes_test)
	FM = fowlkes_mallows_score(classes_real, classes_test)
	AC = accuracy_score(classes_real, classes_test)

	print(f'Rand Index Score:               {RI}')
	print(f'Adjusted Rand Index Score:      {ARI}')
	print(f'Normalized Mutual Index Score:  {NMI}')
	print(f'Adjusted Mutual Info Score:     {AMI}')
	print(f'Fowlkes Mallows Score:          {FM}')
	print(f'Accuracy Score:          \t{AC}')