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