import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('L2-UniFrac/scripts')
import os
import argparse
import L2UniFrac as L2U
import numpy as np
import import itertools as it
import pandas as pd

def get_wgs_L1_pairwise_unifrac(profile_dir, save_as):
    if save_as is None:
        save_as = "pairwise_WGSUniFrac_matrix.csv"
    cur_dir = os.getcwd()
    file_lst = os.listdir(dir)  # list files in the directory
    # print(file_lst)
    os.chdir(dir)
    if '.DS_Store' in file_lst:
        file_lst.remove('.DS_Store')
    sample_lst = [os.path.splitext(profile)[0].split('.')[0] for profile in file_lst] #e.g.env1sam10. i.e.filenames without extension
    #print(sample_lst)
    # enumerate sample_lst, for filling matrix
    id_dict = dict()
    for i, id in enumerate(file_lst):
        id_dict[id] = i
    # initialize matrix
    dim = len(file_lst)
    dist_matrix = np.zeros(shape=(dim, dim))
    count=0
    for pair in it.combinations(file_lst, 2): #all pairwise combinations
        #to keep the running less boring
        count+=1
        if count % 100 == 0:
            print(count)
        id_1, id_2 = pair[0], pair[1]
        i, j = id_dict[id_1], id_dict[id_2]
        profile_list1 = L2U.open_profile_from_tsv(id_1, False)
        profile_list2 = L2U.open_profile_from_tsv(id_2, False)
        name1, metadata1, profile1 = profile_list1[0]
        name2, metadata2, profile2 = profile_list2[0]
        profile1 = L2U.Profile(sample_metadata=metadata1, profile=profile1, branch_length_fun=lambda x: x ** alpha)
        profile2 = L2U.Profile(sample_metadata=metadata2, profile=profile2, branch_length_fun=lambda x: x ** alpha)
        # (Tint, lint, nodes_in_order, nodes_to_index, P, Q) = profile1.make_unifrac_input_no_normalize(profile2)
        (Tint, lint, nodes_in_order, nodes_to_index, P, Q) = profile1.make_unifrac_input_and_normalize(profile2)
        (weighted, _) = L2U.EMDUnifrac_weighted(Tint, lint, nodes_in_order, P, Q)
        dist_matrix[i][j] = dist_matrix[j][i] = weighted
    os.chdir(cur_dir)
    pd.DataFrame(data=dist_matrix, index=sample_lst, columns=sample_lst).to_csv(save_as, sep="\t")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get pairwise unifrac matrix given a directory of profiles.')
    parser.add_argument('-d', '--pdir', type=str, help="Directory of profiles")
    parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")
    args = parser.parse_args()

    get_wgs_L1_pairwise_unifrac(args.pdir, args.save)
