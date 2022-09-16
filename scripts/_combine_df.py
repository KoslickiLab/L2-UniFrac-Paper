import pandas as pd
import os
import argparse

def combine_files(dir, save_as):
    '''
    Combine files in a given directory
    :param dir:
    :return:
    '''
    df_list = []
    for file in os.listdir(dir):
        print(file)
        df = pd.read_csv(file, sep='\t')
        df_list.append(df)
    combined_df = pd.concat(df_list)
    print(combined_df)
    #df.to_csv(save_as, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get pairwise unifrac matrix given a directory of profiles.')
    parser.add_argument('-d', '--dir', type=str, help="Directory of dataframes")
    parser.add_argument('-s', '--save', type=str, help="Save the dataframe file as.")

    args = parser.parse_args()
    combine_files(args.dir, args.save)