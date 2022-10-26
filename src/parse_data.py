import pandas as pd
import numpy as np
from dendropy import Tree, datamodel

def parse_otu_table_no_extend(otu_file, normalize=True):
	'''
	Parses an otu file in tsv format and returns a dict with keys being sample id and values being abundance vector
	:param otu_file: path to an otu table file in .tsv format. Can be converted from .biom format by running
	'biom convert -i table.biom -o table.from_biom.txt --to-tsv'
	:return:
	'''
	df = parse_df(otu_file, header_index=1, index_col='#OTU ID') #remove first row "#Constructed from biom file"
	sample_ids = df.columns.tolist()
	otus = df.index.tolist()
	otus = list(map(lambda x:str(x), otus))
	sample_vector_dict = dict()
	for sample in sample_ids:
		vector = df[sample].tolist()
		if normalize is True:
			vector = vector/np.sum(vector)
		sample_vector_dict[sample] = vector
	return sample_vector_dict, sample_ids, otus

def parse_otu_table(otu_file, nodes_in_order, normalize=True):
	'''
	Parses an otu file in tsv format and returns a dict with keys being sample id and values being abundance vector
	:param otu_file: path to an otu table file in .tsv format. Can be converted from .biom format by running
	'biom convert -i table.biom -o table.from_biom.txt --to-tsv'
	:param Tint:
	:param lint:
	:param nodes_in_order:
	:return:
	'''
	df = parse_df(otu_file, header_index=1, index_col='#OTU ID') #remove first row "#Constructed from biom file"
	sample_ids = df.columns.tolist()
	df.index = df.index.astype("str")
	otus = df.index.tolist()
	print(df.head())
	extended_df = pd.DataFrame(columns=sample_ids, index=nodes_in_order)
	for sample in sample_ids:
		print(sample)
		for otu in otus:
			extended_df[sample][otu] = df[sample][otu]
		if normalize is True:
			extended_df[sample] = extended_df[sample]/np.sum(extended_df[sample])
	extended_df.fillna(0., inplace=True)
	#sample_vector_dict = extended_df.to_dict(orient='list')
	return extended_df

def parse_df(file, header_index=0, index_col=0, sep='\t'):
	df = pd.read_csv(file, sep=sep, header=header_index, index_col=index_col)
	return df