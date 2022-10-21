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
	df = pd.read_table(otu_file, header=1, index_col=0) #remove first row "#Constructed from biom file"
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
	df = pd.read_table(otu_file, header=1, index_col=0) #remove first row "#Constructed from biom file"
	sample_ids = df.columns.tolist()
	print(df.shape)
	otus = df.index.tolist()
	otus = list(map(lambda x: str(x), otus))
	nodes_in_order = list(map(str, nodes_in_order))
	extended_df = pd.DataFrame(columns=sample_ids, index=nodes_in_order)
	for sample in sample_ids:
		print(sample)
		for otu in otus:
			extended_df[sample][otu] = df[sample][otu]
		if normalize is True:
			extended_df[sample] = extended_df[sample]/np.sum(extended_df[sample])
	extended_df.fillna(0., inpoly=True)
	sample_vector_dict = extended_df.to_dict(orient='list')
	return sample_vector_dict, sample_ids

def parse_tree_file(tree_str_file, suppress_internal_node_taxa=True, suppress_leaf_node_taxa=False):
	'''
	Tint,lint,nodes_in_order = parse_tree_file(tree_str_file)
	This function will parse a newick tree file (in the file given by tree_str_file) and return the dictionary of ancestors Tint.
	Tint indexes the nodes by integers, Tint[i] = j means j is the ancestor of i.
	lint is a dictionary returning branch lengths: lint[i,j] = w(i,j) the weight of the edge connecting i and j.
	nodes_in_order is a list of the nodes in the input tree_str such that T[i]=j means nodes_in_order[j] is an ancestor
	of nodes_in_order[i]. Nodes are labeled from the leaves up.
	:param tree_str_file: A tree file in newick format
	'''
	dtree = Tree.get(path=tree_str_file, schema="newick",
							suppress_internal_node_taxa=suppress_internal_node_taxa,
							store_tree_weights=True,
							suppress_leaf_node_taxa = suppress_leaf_node_taxa)
	#Name all the internal nodes
	nodes = dtree.nodes()
	i=0
	for node in nodes:
		if node.taxon == None:
			node.taxon = datamodel.taxonmodel.Taxon(label="temp"+str(i))
			i = i+1
	full_nodes_in_order = [item for item in dtree.levelorder_node_iter()]  # i in path from root to j only if i>j
	full_nodes_in_order.reverse()
	nodes_in_order = [item.taxon.label for item in full_nodes_in_order]  # i in path from root to j only if i>j
	Tint = dict()
	lint = dict()
	nodes_to_index = dict(zip(nodes_in_order, range(len(nodes_in_order))))
	for i in range(len(nodes_in_order)):
		node = full_nodes_in_order[i]
		parent = node.parent_node
		if parent != None:
			Tint[i] = nodes_to_index[parent.taxon.label]
			if isinstance(node.edge.length, float):
				lint[nodes_to_index[node.taxon.label], nodes_to_index[parent.taxon.label]] = node.edge.length
			else:
				lint[nodes_to_index[node.taxon.label], nodes_to_index[parent.taxon.label]] = 0.0
	return Tint,lint,nodes_in_order