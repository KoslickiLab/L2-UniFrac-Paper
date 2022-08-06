import sys, os
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
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import rand_score, accuracy_score
from extract_data import extract_biom_samples, extract_samples, extract_metadata_direct, extract_sample_metadata

class ResNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(9160, 2048)
		self.l2 = nn.Linear(2048, 2048)
		self.l3 = nn.Linear(2048, 5)
		self.do = nn.Dropout(0.1)

	def forward(self, x):
		h1 = nn.functional.relu(self.l1(x))
		h2 = nn.functional.relu(self.l2(h1))
		do = self.do(h2 + h1)
		logits = self.l3(do)
		return logits


if __name__ == '__main__':
	if torch.cuda.is_available():
		model = ResNet().cuda()
	else:
		model = ResNet()

	optimizer = optim.SGD(model.parameters(), lr=1e-2)

	loss = nn.CrossEntropyLoss()

	biom_file = '../data/biom/47422_otu_table.biom'
	metadata_file = '../data/metadata/P_1928_65684500_raw_meta.txt'
	batch_size = 32

	nodes_samples = extract_biom_samples(biom_file)
	sample_ids = extract_samples(biom_file)
	metadata = extract_metadata_direct(metadata_file)
	sample_metadata = extract_sample_metadata(biom_file, metadata_file)

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
	val_loader = []
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
			val_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])
			curr_len = 0
			tmp_x = []
			tmp_y = []

	val_loader.append([torch.FloatTensor([tmp_x]), torch.LongTensor(tmp_y)])

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
		for batch in val_loader:
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
	for batch in val_loader:
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
	for x, y in val_loader:
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