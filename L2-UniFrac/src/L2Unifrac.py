import os

import numpy as np
import matplotlib.pyplot as plt
import dendropy
import sys
import warnings
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
sys.path.append('../tests')
import logging
from collections import defaultdict
import copy

epsilon = sys.float_info.epsilon

# This will return the L2Unifrac distance only
def L2Unifrac_weighted(Tint, lint, nodes_in_order, P, Q, include_tmp_diffab=True):
	'''
	(Z, diffab) = L2Unifrac_weighted(Tint, lint, nodes_in_order, P, Q)
	This function takes the ancestor dictionary Tint, the lengths dictionary lint, the basis nodes_in_order
	and two probability vectors P and Q (typically P = envs_prob_dict[samples[i]], Q = envs_prob_dict[samples[j]]).
	Returns the weighted Unifrac distance Z and the differential abundance. The differential abundance vector diffab 
	is a dictionary with tuple keys using elements of Tint and values diffab[(i, j)] equal to the signed difference 
	of abundance between the two samples restricted to the sub-tree defined by nodes_in_order(i) and weighted by the 
	edge length lint[(i,j)].
	'''
	num_nodes = len(nodes_in_order)
	Z = 0
	diffab = dict()
	partial_sums = P - Q
	for i in range(num_nodes - 1):
		val = partial_sums[i]
		partial_sums[Tint[i]] += val
		if val != 0 and (include_tmp_diffab or nodes_in_order[i][0] != 't'):
			diffab[(i, Tint[i])] = lint[i, Tint[i]]*val # Captures diffab
		Z += lint[i, Tint[i]]*(val**2)
	Z = np.sqrt(Z)
	return (Z, diffab)

def L2Unifrac_weighted_plain(Tint, lint, nodes_in_order, P, Q):
	'''
	Z = L2Unifrac_weighted_plain(ancestors, edge_lengths, nodes_in_order, P, Q)
	This function takes the ancestor dictionary Tint, the lengths dictionary lint, the basis nodes_in_order
	and two probability vectors P and Q (typically P = envs_prob_dict[samples[i]], Q = envs_prob_dict[samples[j]]).
	Returns the weighted Unifrac distance Z.
	'''
	num_nodes = len(nodes_in_order)
	Z = 0
	eps = 1e-8
	partial_sums = P - Q # Vector of partial sums obtained by computing the difference between probabilities of two samples. 
	for i in range(num_nodes - 1):
		val = partial_sums[i]
		if abs(val) > eps:
			partial_sums[Tint[i]] += val
			Z += lint[i, Tint[i]]*(val**2)
	Z = np.sqrt(Z)
	return Z

def push_up(P, Tint, lint, nodes_in_order):
	'''
	P = push_up(P, Tint, lint, nodes_in_order)
	This function takes the ancestor dictionary Tint, the lengths dictionary lint, the basis nodes_in_order
	and the probability vector P.
	Returns the pushed-up probability vector of P with respect to the phylogenetic tree.
	'''
	P_pushed = P + 0  # don't want to stomp on P
	for i in range(len(nodes_in_order) - 1):
		if lint[i, Tint[i]] == 0:
			lint[i, Tint[i]] = epsilon
		P_pushed[Tint[i]] += P_pushed[i]  # push mass up
		P_pushed[i] *= np.sqrt(lint[i, Tint[i]])
	return P_pushed

def build_W2(Tint, lint, nodes_in_order):
	'''
	W2 = build_W2(Tint, lint, nodes_in_order)
	This function takes the ancestor dictionary Tint, the lengths dictionary lint, the basis nodes_in_order.
	Returns the transformation matrix corresponding to the push up operation for use on P probability vectors
	'''
	n = len(nodes_in_order)
	W2 = dok_matrix((n, n), dtype=np.float64)
	for i in range(n):
		cur_node = i
		while cur_node != n:
			if cur_node in Tint:
				W2[cur_node, i] = np.sqrt(lint[cur_node, Tint[cur_node]])
				cur_node = Tint[cur_node]
			else:
				W2[cur_node, i] = 1
				cur_node += 1
	W2 = csr_matrix(W2)
	return W2

def inverse_push_up(P, Tint, lint, nodes_in_order):
	'''
	P = inverse_push_up(P, Tint, lint, nodes_in_order)
	This function takes the ancestor dictionary Tint, the lengths dictionary lint, the basis nodes_in_order
	and the pushed-up probability vector P.
	Returns the probability vector of P with respect to the phylogenetic tree.
	'''
	P_pushed = np.zeros(P.shape)  # don't want to stomp on P
	for i in range(len(nodes_in_order) - 1):
		if lint[i, Tint[i]] == 0:
			edge_length = epsilon
		else:
			edge_length = lint[i, Tint[i]]
		p_val = P[i]
		P_pushed[i] += 1/np.sqrt(edge_length) * p_val  # re-adjust edge lengths
		if P_pushed[i] < epsilon:
			P_pushed[i] = 0
		P_pushed[Tint[i]] -= 1/np.sqrt(edge_length) * p_val  # propagate mass upward, via subtraction, only using immediate descendants
	root = len(nodes_in_order) - 1
	P_pushed[root] += P[root]
	return P_pushed

def inverse_W2(W2):
	'''
	:param W2: an nxn matrix of edge lengths on a tree, where the diagonal corresponds to weights of each node n and all descendents j 
	of n are assigned the same weight at position j in row n.
	:return: the inverse of matrix W2
	'''
	return inv(W2)

def mean_of_vectors(L):
	'''
	:param L: a list of vectors
	:return: a vector with each entry i being the mean of vectors of L at position i
	'''
	return np.mean(L, axis=0)

def plot_diffab(nodes_in_order, taxonomy_in_order, diffab, P_label, Q_label, plot_zeros=True, thresh=0, show=True, maxDisp=0, includeTemp=True):
	'''
	plot_diffab(nodes_in_order, diffab, P_label, Q_label)
	Plots the differential abundance vector.
	:param nodes_in_order: list returned from parse_envs
	:param diffab: differential abundance vector (returned from one flavor of L2Unifrac)
	:param P_label: label corresponding to the sample name for P (e.g. when calling L2Unifrac_weighted(Tint, lint, nodes_in_order, P, Q))
	:param Q_label: label corresponding to the sample name for P (e.g. when calling L2Unifrac_weighted(Tint, lint, nodes_in_order, P, Q))
	:param plot_zeros: flag (either True or False) that specifies if the zero locations should be plotted. Warning, if your tree is large and plot_zeros=True, this can cause a crash.
	:param thresh: only plot those parts of the diffab vector that are above thresh, specify everything else as zero
	:return: None (makes plot)
	'''
	new_tax_in_order = []
	for i in range(len(taxonomy_in_order)):
		new_tax_in_order.append(taxonomy_in_order[i].split(';')[-2:-1][0])

	x = range(len(nodes_in_order))
	y = np.zeros(len(nodes_in_order))
	keys = diffab.keys()
	for key in keys:
		y[key[0]] = diffab[key]

	pos_loc = [x[i] for i in range(len(y)) if (y[i] > thresh and 'temp' not in nodes_in_order[i]) or (y[i] > thresh and includeTemp)]
	neg_loc = [x[i] for i in range(len(y)) if (y[i] < -thresh and 'temp' not in nodes_in_order[i]) or (y[i] < -thresh and includeTemp)]
	zero_loc = [x[i] for i in range(len(y)) if (-thresh <= y[i] <= thresh and 'temp' not in nodes_in_order[i]) or (-thresh <= y[i] <= thresh and includeTemp)]

	pos_val = [y[i] for i in range(len(y)) if (y[i] > thresh and 'temp' not in nodes_in_order[i]) or (y[i] > thresh and includeTemp)]
	neg_val = [y[i] for i in range(len(y)) if (y[i] < -thresh and 'temp' not in nodes_in_order[i]) or (y[i] < -thresh and includeTemp)]
	zero_val = [y[i] for i in range(len(y)) if (-thresh <= y[i] <= thresh and 'temp' not in nodes_in_order[i]) or (-thresh <= y[i] <= thresh and includeTemp)]

	# Increase threshold until pos and neg are less than the max display (very inefficient... TODO: optimize using by taking top 10 or so elements directly)
	while True:
		if (len(pos_val) > maxDisp or len(neg_val) > maxDisp) and maxDisp > 0:
			thresh *= 1.05
		else:
			break

		if len(pos_val) > maxDisp:
			pos_loc = [x[i] for i in range(len(y)) if (y[i] > thresh and 'temp' not in nodes_in_order[i]) or (y[i] > thresh and includeTemp)]
		if len(neg_val) > maxDisp:
			neg_loc = [x[i] for i in range(len(y)) if (y[i] < -thresh and 'temp' not in nodes_in_order[i]) or (y[i] < -thresh and includeTemp)]

		if len(pos_val) > maxDisp:
			pos_val = [y[i] for i in range(len(y)) if (y[i] > thresh and 'temp' not in nodes_in_order[i]) or (y[i] > thresh and includeTemp)]
		if len(neg_val) > maxDisp:
			neg_val = [y[i] for i in range(len(y)) if (y[i] < -thresh and 'temp' not in nodes_in_order[i]) or (y[i] < -thresh and includeTemp)]

	if not pos_loc:
		raise Exception('Threshold too high or max too low! Please change and try again.')
	if not neg_loc:
		raise Exception('Threshold too high or max too low! Please change and try again.')

	# The following is to get the indicies in order. Basically, I iterate down both pos_loc and neg_loc simultaneously
	# and create new lists (pos_loc_adj and neg_loc_adj) that are in the same order as pos_loc and neg_loc, but whose
	# union of indicies is equal to range(len(pos_loc + neg_loc)). Simply to make things pretty
	if plot_zeros:
		pos_loc_adj = pos_loc
		neg_loc_adj = neg_loc
		zero_loc_adj = zero_loc
	else:
		pos_loc_adj = []
		neg_loc_adj = []
		tick_names = []

		# rename the indicies so they are increasing by 1
		pos_ind = 0
		neg_ind = 0
		it = 0
		while pos_ind < len(pos_loc) or neg_ind < len(neg_loc):
			if pos_ind >= len(pos_loc):
				neg_loc_adj.append(it)
				tick_names.append(new_tax_in_order[neg_loc[neg_ind]])
				it += 1
				neg_ind += 1
			elif neg_ind >= len(neg_loc):
				pos_loc_adj.append(it)
				tick_names.append(new_tax_in_order[pos_loc[pos_ind]])
				it += 1
				pos_ind += 1
			elif pos_loc[pos_ind] < neg_loc[neg_ind]:
				pos_loc_adj.append(it)
				tick_names.append(new_tax_in_order[pos_loc[pos_ind]])
				it += 1
				pos_ind += 1
			elif pos_loc[pos_ind] > neg_loc[neg_ind]:
				neg_loc_adj.append(it)
				tick_names.append(new_tax_in_order[neg_loc[neg_ind]])
				it += 1
				neg_ind +=1
			else:
				print('Something went wrong')
				break


	fig, ax = plt.subplots()

	markerline, stemlines, baseline = ax.stem(neg_loc_adj, neg_val)
	plt.setp(baseline, linewidth=1, color='k')
	plt.setp(markerline, color='r')
	plt.setp(stemlines, linewidth=3, color='r')

	markerline, stemlines, baseline = ax.stem(pos_loc_adj, pos_val)
	plt.setp(baseline, linewidth=1, color='k')
	plt.setp(markerline, color='b')
	plt.setp(stemlines, linewidth=3, color='b')

	if plot_zeros:
		markerline, stemlines, baseline = ax.stem(zero_loc, zero_val)
		plt.setp(baseline, linewidth=1, color='k')
		plt.setp(markerline, color='k')
		plt.setp(stemlines, linewidth=3, color='k')

	plt.ylabel('DiffAbund', fontsize=16)
	plt.gcf().subplots_adjust(right=0.93, left=0.15)

	# If you want the zeros plotted, label EVERYTHING, otherwise just label the things that are there...
	if plot_zeros:
		plt.xticks(x, nodes_in_order, rotation='vertical', fontsize=8)
	else:
		plt.xticks(range(len(pos_loc_adj + neg_loc_adj)), tick_names, rotation='vertical', fontsize=8)

	plt.subplots_adjust(bottom=0.35, top=.93)
	plt.text(plt.xticks()[0][-1]+0.1, max(pos_val), P_label, rotation=90, horizontalalignment='center', verticalalignment='top', multialignment='center', color='b', fontsize=14)
	plt.text(plt.xticks()[0][-1]+0.1, min(neg_val), Q_label, rotation=90, horizontalalignment='center', verticalalignment='bottom', multialignment='center', color='r', fontsize=14)
	
	if show:
		plt.show()
	else:
		return fig

#WGS parts
#input profile.
class Prediction:
    def __init__(self):
        pass

    @property
    def rank(self):
        return self.__rank

    @property
    def taxid(self):
        return self.__taxid

    @property
    def percentage(self):
        return self.__percentage

    @property
    def taxpath(self):
        return self.__taxpath

    @property
    def taxpathsn(self):
        return self.__taxpathsn

    @rank.setter
    def rank(self, rank):
        self.__rank = rank

    @taxid.setter
    def taxid(self, taxid):
        self.__taxid = taxid

    @percentage.setter
    def percentage(self, percentage):
        self.__percentage = percentage

    @taxpath.setter
    def taxpath(self, taxpath):
        self.__taxpath = taxpath

    @taxpathsn.setter
    def taxpathsn(self, taxpathsn):
        self.__taxpathsn = taxpathsn

    def get_dict(self):
        return self.__dict__

    def get_pretty_dict(self):
        return {property.split("_")[3]: value for property, value in self.__dict__.items()}

    def get_metadata(self):
        return {'rank': self.__rank, 'taxpath': self.__taxpath, 'taxpathsn': self.__taxpathsn}

class Profile(object):
    def __init__(self, sample_metadata=None, profile=None, branch_length_fun=lambda x: 1 / x):
        self.sample_metadata = sample_metadata
        self.profile = profile
        self._data = dict()
        # Stick in the root node just to make sure everything is consistent
        self._data["-1"] = dict()
        self._data["-1"]["rank"] = None
        self._data["-1"]["tax_path"] = list()
        self._data["-1"]["tax_path_sn"] = list()
        self._data["-1"]["abundance"] = 0
        self._data["-1"]["descendants"] = list()
        self._header = list()
        self._tax_id_pos = None
        self._rank_pos = None
        self._tax_path_pos = None
        self._tax_path_sn_pos = None
        self._abundance_pos = None
        self._eps = .0000000000000001  # This is to act like zero, ignore any lines with abundance below this quantity
        self._all_keys = ["-1"]
        self._merged_flag = False
        self.root_len = 1  # the length you want between the "root" of "-1" and the superkingdom level (eg. Bacteria)
        self.branch_len_func = branch_length_fun  # Given a node n at depth d in the tree, branch_len_func(d)
        # is how long you want the branch length between n and ancestor(n) to be
        self._data["-1"]["branch_length"] = self.root_len
        self.parse_file()  # TODO: this sets all the branch lengths to 1 currentlclass Profile(object):


    def parse_file(self):
        _data = self._data
        _all_keys = self._all_keys
        _header = self._header
        for k, v in self.sample_metadata.items():
            _header.append('{}:{}'.format(k, v))

        # populate all the correct keys
        for prediction in self.profile:
            _all_keys.append(prediction.taxid.strip())

        # crawl over all profiles tax_path and create the ancestors and descendants list
        for prediction in self.profile:
            tax_id = prediction.taxid.strip()
            tax_path = prediction.taxpath.strip().split("|")  # this will be a list, join up late
            if tax_id not in _data:
                _data[tax_id] = dict()
            else:
                raise Exception(f"Improperly formatted profile: row starting with {tax_id} shows up more than once")
            _data[tax_id]["tax_path"] = tax_path

            # populate abundance
            _data[tax_id]["abundance"] = prediction.percentage

            # populate tax path sn
            if not (prediction.taxpathsn is None):  # might not be present
                _data[tax_id]["tax_path_sn"] = prediction.taxpathsn.strip().split(
                    "|")  # this will be a list, join up later

            # populate the rank
            _data[tax_id]["rank"] = prediction.rank.strip()

            # populate the branch length
            _data[tax_id]["branch_length"] = self.tax_path_to_branch_len(tax_path, self.branch_len_func, self.root_len)

            # Find the ancestors
            if len(tax_path) <= 1:  # note, due to the format, we will never run into the case tax_path == []
                _data[tax_id]["ancestor"] = "-1"  # no ancestor, it's a root
            else:  # go from the bottom up, looking for an ancestor that is an acceptable key
                ancestor = "-1"  # this is the default
                tax_path_rev = tax_path[::-1]
                for potential_ancestor in tax_path_rev:
                    if potential_ancestor != tax_id and potential_ancestor in _all_keys:
                        ancestor = potential_ancestor
                        break  # you found the ancestor, so can quit looking
                _data[tax_id]["ancestor"] = ancestor

            # Create a placeholder descendant key initialized to [], just so each tax_id has a descendant key associated to it
            if "descendants" not in _data[tax_id]:  # if this tax_id doesn't have a descendant list,
                _data[tax_id]["descendants"] = list()  # initialize to empty list

        self._add_descendants()
        self._delete_missing()  # make sure there aren't any missing internal nodes

    def _add_descendants(self):
        """
        Idea here is to look at all the ancestors of each key, and make the key the descendant of that ancestor
        Returns
        -------
        None: modifies Profile in place
        """
        _data = self._data
        _all_keys = self._all_keys
        for prediction in self.profile:
            tax_id = prediction.taxid.strip()  # the tax ID we are looking at
            ancestor = _data[tax_id]['ancestor']  # the tax ID's ancestor
            if tax_id not in _data[ancestor]['descendants']:
                _data[ancestor]['descendants'].append(
                    tax_id)  # so make the tax ID we're looking at the descendant of the ancestor

    def _delete_missing(self):
        """
        Deletes from the descendants all those taxids that aren't keys in the profile (i.e. there is no line that starts with that taxID)
        Returns
        -------
        none: modifies Profile in place
        """
        for key in self._data:
            clean_descendants = []
            for descendant in self._data[key]["descendants"]:
                if descendant in self._all_keys:  # if it's one of the taxids that the line starts with, add it
                    clean_descendants.append(descendant)
                else:
                    pass  # don't include the taxids that aren't actually in the final tax tree
            self._data[key]["descendants"] = clean_descendants
        return

    def write_file(self, out_file_name=None):
        if out_file_name is None:
            raise Exception
        _data = self._data
        keys = _data.keys()
        # This will be annoying to keep things in order...
        # Let's iterate on the length of the tax_path since we know that will be in there
        tax_path_lengths = max([len(_data[key]["tax_path"]) for key in keys])
        fid = open(out_file_name, 'w')
        # Write the header
        for head in self._header:
            fid.write("%s\n" % head)

        # Loop over length of tax_path and write data
        # always make the output tax_id, rank, tax_path, tax_path_sn, abundance in that order
        for path_length in range(1, tax_path_lengths + 1):
            for key in keys:
                if len(_data[key]["tax_path"]) == path_length and _data[key]["abundance"] > self._eps:
                    line_data = _data[key]
                    fid.write("%s\t" % key)
                    if self._rank_pos is not None:
                        fid.write("%s\t" % line_data["rank"])
                    fid.write("%s\t" % "|".join(line_data["tax_path"]))
                    if self._tax_path_sn_pos is not None:
                        fid.write("%s\t" % "|".join(line_data["tax_path_sn"]))
                    fid.write("%f\n" % line_data["abundance"])
        fid.close()
        return

    def threshold(self, threshold=None):
        if threshold is None:
            raise Exception
        _data = self._data
        keys = _data.keys()
        for key in keys:
            if _data[key]["abundance"] < threshold:
                _data[key]["abundance"] = 0
        return

    def _subtract_down(self):
        # helper function to push all the weights up by subtracting
        # NOTE: when subtracting, need to start at root and go down
        # NOTE: when adding, need to start at leaves and go up
        _data = self._data
        keys = _data.keys()
        # This will be annoying to keep things in order...
        # Let's iterate on the length of the tax_path since we know that will be in there
        tax_path_lengths = max([len(_data[key]["tax_path"]) for key in keys])
        for path_length in range(1, tax_path_lengths):  # eg tax_path_lengths = 5, use 1,2,3,4 since we stop at leaves
            for key in keys:
                if len(_data[key]["tax_path"]) == path_length:
                    descendants = _data[key]["descendants"]  # get all descendants
                    for descendant in descendants:
                        _data[key]["abundance"] -= _data[descendant]["abundance"]  # subtract the descendants abundance

    def _add_up(self):
        # helper function to push all the weights up by subtracting
        # NOTE: when subtracting, need to start at root and go down
        # NOTE: when adding, need to start at leaves and go up
        _data = self._data
        keys = _data.keys()
        # This will be annoying to keep things in order...
        # Let's iterate on the length of the tax_path since we know that will be in there
        tax_path_lengths = max([len(_data[key]["tax_path"]) for key in keys])
        for path_length in range(tax_path_lengths, 1,
                                 -1):  # eg tax_path_lengths = 5, use 5,4,3,2, since we stop at roots
            for key in keys:
                if len(_data[key]["tax_path"]) == path_length:
                    ancestor = _data[key]["ancestor"]
                    if ancestor in _data:  # don't do anything if this is a/the root node
                        _data[ancestor]["abundance"] += _data[key]["abundance"]  # add the descendants abundance

    def normalize(self):
        # Need to really push it up while subtracting, then normalize, then push up wile adding
        # self._push_up(operation="subtract")
        self._subtract_down()
        _data = self._data
        keys = _data.keys()
        total_abundance = 0
        for key in keys:
            total_abundance += _data[key]["abundance"]
        # print(total_abundance)
        for key in keys:
            if total_abundance > 0:
                _data[key]["abundance"] /= total_abundance
                _data[key]["abundance"] *= 100  # make back into a percentage
        # self._push_up(operation="add")
        self._add_up()
        return

    def merge(self, other):
        # Warning: not checking for taxonomic consistency
        if not isinstance(other, Profile):
            print("Only works with other Profiles")
            raise Exception
        if self._merged_flag is False:
            self._header.insert(0, "# This is a merged file, ignore files in headers below")
            self._merged_flag = True
        _data = self._data
        _other_data = other._data
        other_keys = _other_data.keys()
        for key in other_keys:
            if key in _data:
                _data[key]["abundance"] += _other_data[key]["abundance"]  # if already in there, add abundances
            else:
                _data[key] = copy.copy(_other_data[key])  # otherwise use the whole thing

    @staticmethod
    def tax_path_to_branch_len(tax_path, func, root_len=1):
        """
        This function modifies the branch lengths based on the input tax_path.
        intent is: ["2", "", "123", "456"] would result in a branch length of func(4)
        Parameters
        ----------
        tax_path : a list of strings (tax ID's)
        func : a function whose argument is the depth in the tree of a tax ID, and whose output is the branch length
               from the tax ID to its ancestor.
        root_len : how long you want the root of the tree "-1" to be to the descendants (eg. "-1" -> "Bacteria")
        Returns
        -------
        float
        """
        # eg. "-1" -> "Bacteria" should have a branch length of root_len
        if not tax_path:
            return root_len
        else:
            depth_in_tree = len(tax_path)  # this takes into account that the tax_path doesn't include the root of "-1"
            return func(depth_in_tree)

    def make_unifrac_input_and_normalize(self):
        _data = self._data
        _data_keys = _data.keys()
        tax_path_lengths = max([len(_data[key]["tax_path"]) for key in _data_keys])
        all_keys = set(_data_keys)
        nodes_in_order = []
        for path_length in range(tax_path_lengths, 0, -1):
            for key in all_keys:
                if key in _data:
                    if len(_data[key]["tax_path"]) == path_length:
                        if key not in nodes_in_order:
                            nodes_in_order.append(key)
        # Make the graph
        # Put the root at the very end
        if '-1' in nodes_in_order:
            nodes_in_order.pop(nodes_in_order.index('-1'))
            nodes_in_order.append('-1')
        else:
            nodes_in_order.append('-1')
        Tint = dict()
        lint = dict()
        for key in nodes_in_order:
            if key in _data:
                if "ancestor" in _data[key]:  # If ancestor is not in there, then it's an ancestor
                    ancestor = _data[key]["ancestor"]
                    Tint[key] = ancestor
                    lint[key, ancestor] = _data[key]["branch_length"]
        nodes_to_index = dict(zip(nodes_in_order, range(len(nodes_in_order))))  # maps '45202.15' -> 0 (i.e taxID to integer index)

        # Now need to change over to the integer-based indexing
        Tint2 = dict()
        lint2 = dict()
        nodes_in_order2 = []
        for key in nodes_in_order:
            if key in Tint:
                ancestor = Tint[key]
                Tint2[nodes_to_index[key]] = nodes_to_index[ancestor]
                if (key, ancestor) in lint:
                    lint2[nodes_to_index[key], nodes_to_index[ancestor]] = lint[key, ancestor]
            nodes_in_order2.append(nodes_to_index[key])

        # Next make the probability distributions
        # Would be nice if I could find a non-destructive way to subtract up and normalize

        # Do it for P
        self._subtract_down()
        keys = _data.keys()
        total_abundance = 0
        for key in keys:
            total_abundance += _data[key]["abundance"]
        # print(total_abundance)
        for key in keys:
            if total_abundance > 0:
                _data[key]["abundance"] /= total_abundance  # Should be a fraction, summing to 1
        P = np.zeros(len(nodes_in_order))
        for key_ind in range(len(nodes_in_order)):
            key = nodes_in_order[key_ind]
            if key in _data:
                P[key_ind] = _data[key]["abundance"]

        # Make back into percentages and add the mass back up (effectively normalizing the vector)
        for key in keys:
            if total_abundance > 0:
                _data[key]["abundance"] *= 100
        self._add_up()

        return Tint2, lint2, nodes_in_order2, nodes_to_index, P

    def make_unifrac_input_no_normalize(self, other):
        if not isinstance(other, Profile):
            raise Exception
        _data = self._data
        _other_data = other._data

        _data_keys = _data.keys()
        tax_path_lengths1 = max([len(_data[key]["tax_path"]) for key in _data_keys])
        _other_data_keys = _other_data.keys()
        tax_path_lengths2 = max([len(_other_data[key]["tax_path"]) for key in _other_data_keys])
        tax_path_lengths = max(tax_path_lengths1, tax_path_lengths2)
        all_keys = set(_data_keys)
        all_keys.update(_other_data_keys)  # all the taxID's in the union of self and other profile
        nodes_in_order = []
        for path_length in range(tax_path_lengths, 0, -1):
            for key in all_keys:
                if key in _data:
                    if len(_data[key]["tax_path"]) == path_length:
                        if key not in nodes_in_order:
                            nodes_in_order.append(key)
                elif key in _other_data:
                    if len(_other_data[key]["tax_path"]) == path_length:
                        if key not in nodes_in_order:
                            nodes_in_order.append(key)
        # Make the graph
        # Put the root at the very end
        if '-1' in nodes_in_order:
            nodes_in_order.pop(nodes_in_order.index('-1'))
            nodes_in_order.append('-1')
        else:
            nodes_in_order.append('-1')
        Tint = dict()
        lint = dict()
        for key in nodes_in_order:
            if key in _data:
                if "ancestor" in _data[key]:  # If ancestor is not in there, then it's an ancestor
                    ancestor = _data[key]["ancestor"]
                    Tint[key] = ancestor
                    lint[key, ancestor] = _data[key]["branch_length"]
            elif key in _other_data:
                if "ancestor" in _other_data[key]:
                    ancestor = _other_data[key]["ancestor"]
                    Tint[key] = ancestor
                    lint[key, ancestor] = _other_data[key]["branch_length"]
        nodes_to_index = dict(
            zip(nodes_in_order, range(len(nodes_in_order))))  # maps '45202.15' -> 0 (i.e taxID to integer index)

        # Now need to change over to the integer-based indexing
        Tint2 = dict()
        lint2 = dict()
        nodes_in_order2 = []
        for key in nodes_in_order:
            if key in Tint:
                ancestor = Tint[key]
                Tint2[nodes_to_index[key]] = nodes_to_index[ancestor]
                if (key, ancestor) in lint:
                    lint2[nodes_to_index[key], nodes_to_index[ancestor]] = lint[key, ancestor]
            nodes_in_order2.append(nodes_to_index[key])

        # Next make the probability distributions
        # Would be nice if I could find a non-destructive way to subtract up and normalize

        # Do it for P
        self._subtract_down()
        keys = _data.keys()
        total_abundance = 0
        for key in keys:
            total_abundance += _data[key]["abundance"]
        # print(total_abundance)
        for key in keys:
            if total_abundance > 0:
                # _data[key]["abundance"] /= total_abundance  # Should be a fraction, summing to 1
                pass
        P = np.zeros(len(nodes_in_order))
        for key_ind in range(len(nodes_in_order)):
            key = nodes_in_order[key_ind]
            if key in _data:
                P[key_ind] = _data[key]["abundance"]

        # Make back into percentages and add the mass back up (effectively normalizing the vector)
        # for key in keys:
        #    if total_abundance > 0:
        #        _data[key]["abundance"] *= 100
        self._add_up()

        # Next do for Q
        other._subtract_down()
        keys = _other_data.keys()
        total_abundance = 0
        for key in keys:
            total_abundance += _other_data[key]["abundance"]
        # print(total_abundance)
        for key in keys:
            if total_abundance > 0:
                # _other_data[key]["abundance"] /= total_abundance  # should be a fraction, summing to 1
                pass
        Q = np.zeros(len(nodes_in_order))
        for key_ind in range(len(nodes_in_order)):
            key = nodes_in_order[key_ind]
            if key in _other_data:
                Q[key_ind] = _other_data[key]["abundance"]

        # Make back into percentages and add the mass back up (effectively normalizing the vector)
        # for key in keys:
        #    if total_abundance > 0:
        #        _other_data[key]["abundance"] *= 100
        other._add_up()

        return Tint2, lint2, nodes_in_order2, nodes_to_index, P / 100., Q / 100.

def open_profile_from_tsv(file_path, normalize):
    header = {}
    column_name_to_index = {}
    profile = []
    samples_list = []
    predictions_dict = {}
    reading_data = False
    got_column_indices = False

    with open(file_path) as read_handler:
        for line in read_handler:
            if len(line.strip()) == 0 or line.startswith("#"):
                continue
            line = line.rstrip('\n')

            # parse header with column indices
            if line.startswith("@@"):
                for index, column_name in enumerate(line[2:].split('\t')):
                    column_name_to_index[column_name] = index
                index_rank, index_taxid, index_percentage, index_taxpath, index_taxpathsn = get_column_indices(column_name_to_index)
                got_column_indices = True
                reading_data = False
                continue

            # parse header with metadata
            if line.startswith("@"):
                # if last line contained sample data and new header starts, store profile for sample
                if reading_data:
                    if 'SAMPLEID' in header and 'VERSION' in header and 'RANKS' in header:
                        if len(profile) > 0:
                            samples_list.append((header['SAMPLEID'], header, profile))
                            profile = []
                            predictions_dict = {}
                    else:
                        logging.getLogger('opal').critical(
                            "Header in file {} is incomplete. Check if the header of each sample contains at least SAMPLEID, VERSION, and RANKS.\n".format(
                                file_path))
                        raise RuntimeError
                    header = {}
                reading_data = False
                got_column_indices = False
                key, value = line[1:].split(':', 1)
                header[key.upper()] = value.strip()
                continue

            if not got_column_indices:
                logging.getLogger('opal').critical(
                    "Header line starting with @@ in file {} is missing or at wrong position.\n".format(file_path))
                raise RuntimeError

            reading_data = True
            row_data = line.split('\t')

            taxid = row_data[index_taxid]
            # if there is already a prediction for taxon, only sum abundance
            if taxid in predictions_dict:
                prediction = predictions_dict[taxid]
                prediction.percentage += float(row_data[index_percentage])
            else:
                if float(row_data[index_percentage]) == .0:
                    continue
                prediction = Prediction()
                predictions_dict[taxid] = prediction
                prediction.taxid = row_data[index_taxid]
                prediction.rank = row_data[index_rank]
                prediction.percentage = float(row_data[index_percentage])
                prediction.taxpath = row_data[index_taxpath]
                if isinstance(index_taxpathsn, int):
                    prediction.taxpathsn = row_data[index_taxpathsn]
                else:
                    prediction.taxpathsn = None
                profile.append(prediction)

    # store profile for last sample
    if 'SAMPLEID' in header and 'VERSION' in header and 'RANKS' in header:
        if reading_data and len(profile) > 0:
            samples_list.append((header['SAMPLEID'], header, profile))
    else:
        logging.getLogger('opal').critical(
            "Header in file {} is incomplete. Check if the header of each sample contains at least SAMPLEID, VERSION, and RANKS.\n".format(
                file_path))
        raise RuntimeError

    if normalize:
        normalize_samples(samples_list)

    return samples_list

def get_column_indices(column_name_to_index):
    if "TAXID" not in column_name_to_index:
        logging.getLogger('opal').critical("Column not found: {}".format("TAXID"))
        raise RuntimeError
    if "RANK" not in column_name_to_index:
        logging.getLogger('opal').critical("Column not found: {}".format("RANK"))
        raise RuntimeError
    if "PERCENTAGE" not in column_name_to_index:
        logging.getLogger('opal').critical("Column not found: {}".format("PERCENTAGE"))
        raise RuntimeError
    if "TAXPATH" not in column_name_to_index:
        logging.getLogger('opal').critical("Column not found: {}".format("TAXPATH"))
        raise RuntimeError
    index_taxid = column_name_to_index["TAXID"]
    index_rank = column_name_to_index["RANK"]
    index_percentage = column_name_to_index["PERCENTAGE"]
    index_taxpath = column_name_to_index["TAXPATH"]
    if "TAXPATHSN" in column_name_to_index:
        index_taxpathsn = column_name_to_index["TAXPATHSN"]
    else:
        index_taxpathsn = None
    return index_rank, index_taxid, index_percentage, index_taxpath, index_taxpathsn

def normalize_samples(samples_list):
    for sample in samples_list:
        sample_id, sample_metadata, profile = sample
        sum_per_rank = defaultdict(float)
        for prediction in profile:
            sum_per_rank[prediction.rank] += prediction.percentage
        for prediction in profile:
            if prediction.percentage > 0:
                prediction.percentage = (prediction.percentage / sum_per_rank[prediction.rank]) * 100.0

def push_up_from_wgs_profile(profile_file, branch_length_fun=lambda x: 1/x):
    profile_list = open_profile_from_tsv(profile_file, False)
    name,metadata, profile = profile_list[0]
    profile = Profile(sample_metadata=metadata, profile=profile, branch_length_fun=branch_length_fun)
    (Tint, lint, nodes_in_order, nodes_to_index, P) = profile.make_unifrac_input_and_normalize()
    pushed_up_P = push_up(P, Tint, lint, nodes_in_order)
    return pushed_up_P

def parse_otu_table(otu_table_file):
    '''
    Takes in a file with multiple samples and returns the samples represented in a list
    :param otu_table_file: A file with rows being otu/taxids and columns being samples
    :return: A list of vectors
    '''
    return

def merge_profiles_by_dir(dir, branch_length_fun=lambda x:1/x):
    '''
    Take in a directory containin profiles, standardize the length of the distribution vectors represented by the profiles
    :param dir: a directory containing profiles
    :return: a dict of samples, i.e. {sample_id: vector}, and taxids in order
    '''
    file_lst = [f for f in os.listdir(dir) if f.endswith('.profile')]
    os.chdir(dir)
    all_taxids = set()
    sample_dict = dict()
    for file in file_lst:
        profile_list = open_profile_from_tsv(file, False)
        name, metadata, profile = profile_list[0]
        profile = Profile(sample_metadata=metadata, profile=profile, branch_length_fun=branch_length_fun)
        #print(profile.sample_metadata['SAMPLEID'])
        #if profile.sample_metadata['SAMPLEID'] == 'unnamed sample':
        #    profile.sample_metadata['SAMPLEID'] = sample_id
        #count total number of taxids
        for prediction in profile.profile:
            #print(prediction.taxid)
            all_taxids.add(prediction.taxid)
            #print(prediction.percentage)
    print("Total number of organisms:", len(all_taxids))
    all_taxids_list = list(all_taxids)
    for file in file_lst:
        sample_id = os.path.splitext(file)[0].split('.')[0]
        profile_list = open_profile_from_tsv(file, False)
        name, metadata, profile = profile_list[0]
        profile = Profile(sample_metadata=metadata, profile=profile, branch_length_fun=branch_length_fun)
        taxid_list = [prediction.taxid for prediction in profile.profile]
        abundance_list = [prediction.percentage for prediction in profile.profile]
        #print(taxid_list)
        #print(abundance_list)
        tax_abund_dict = dict(zip(taxid_list, abundance_list))
        distribution_vector = [0.] * len(all_taxids_list)
        for i, tax in enumerate(all_taxids_list):
            if tax in taxid_list: #if this sample contains this taxid
                distribution_vector[i] = tax_abund_dict[tax]
        sample_dict[sample_id] = distribution_vector
    #print(sample_dict)
    return sample_dict, all_taxids_list


#def run_tests():
#	import test_meanUnifrac as test
#	test.run_tests()


#if __name__ == '__main__':
#	run_tests()