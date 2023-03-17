import os
import itertools as it
import numpy as np
import sys

import pandas as pd

epsilon = sys.float_info.epsilon

def EMDUnifrac_weighted(Tint, lint, nodes_in_order, P, Q):
    '''
    (Z, diffab) = EMDUnifrac_weighted(Tint, lint, nodes_in_order, P, Q)
    This function takes the ancestor dictionary Tint, the lengths dictionary lint, the basis nodes_in_order
    and two probability vectors P and Q (typically P = envs_prob_dict[samples[i]], Q = envs_prob_dict[samples[j]]).
    Returns the weighted Unifrac distance Z and the flow F. The flow F is a dictionary with keys of the form (i,j) where
    F[(i,j)] == num means that in the calculation of the Unifrac distance, a total mass of num was moved from the node
    nodes_in_order[i] to the node nodes_in_order[j].
    '''
    num_nodes = len(nodes_in_order)
    Z = 0
    partial_sums = P - Q
    for i in range(num_nodes - 1):
        val = partial_sums[i]
        partial_sums[Tint[i]] += val
        Z += lint[i, Tint[i]] * abs(val)
    return Z

def pairwise_L1EMDUniFrac_weighted(sample_dict, Tint, lint, nodes_in_order):
    df = pd.DataFrame(columns=list(sample_dict.keys()), index=list(sample_dict.keys()))
    print(sample_dict.keys())
    print('sample_dict length in L1EMDUniFrac function', len(sample_dict))
    for i in df.columns:
        df[i][i] = 0.
    for pair in it.combinations(sample_dict.keys(), 2): #all pairwise combinations
        sample1, sample2 = pair[0], pair[1]
        P, Q = sample_dict[sample1], sample_dict[sample2]
        unifrac = EMDUnifrac_weighted(Tint, lint, nodes_in_order, P, Q)
        df[sample1][sample2] = df[sample2][sample1] = unifrac
    return df

def push_up(P, Tint, lint, nodes_in_order):
    P_pushed = P
    for i in range(len(nodes_in_order) - 1):
        if lint[i, Tint[i]] == 0:
            lint[i, Tint[i]] = np.epsilon
        P_pushed[Tint[i]] += P_pushed[i]  # push mass up
        P_pushed[i] *= lint[i, Tint[i]]  # multiply mass at this node by edge length above it
    return P_pushed

def median_of_vectors(L):
    '''
    :param L: a list of vectors
    :return: a vector with each entry i being the median of vectors of L at position i
    '''
    return np.median(L, axis=0)