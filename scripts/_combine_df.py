import pandas as pd
import os
import argparse

def combine_files(dir, save_as):
    '''
    Combine files in a given directory
    :param dir:
    :return:
    '''
    cur_dir = os.getcwd()
    df_list = []
    files = os.listdir(dir)
    os.chdir(dir)
    for file in files:
        print(file)
        df = pd.read_csv(file, sep='\t')
        df_list.append(df)
    combined_df = pd.concat(df_list)
    print(combined_df)
    os.chdir(cur_dir)
    combined_df.to_csv(save_as, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get pairwise unifrac matrix given a directory of profiles.')
    parser.add_argument('-d', '--dir', type=str, help="Directory of dataframes")
    parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")

    args = parser.parse_args()
    combine_files(args.dir, args.save)