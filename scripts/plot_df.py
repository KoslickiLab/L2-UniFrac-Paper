import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from skbio import DistanceMatrix #to install: pip install scikit-bio

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
except:
    pass
from src.helper import get_pcoa

def main():
    parser = argparse.ArgumentParser(description="Get plot from a dataframe.")
    parser.add_argument('-f', '--file', type=str, help="Dataframe file.")
    parser.add_argument('-x', '--x', type=str, help="x axis.")
    parser.add_argument('-y', '--y', type=str, help="y axis.")
    parser.add_argument('-s', '--save', type=str, help="If wants to save df for future use, file name to save as")
    parser.add_argument('-hue', '--hue', type=str, help="Hue")
    parser.add_argument('-t', '--type', type=str, help="Plot type.", choices=['box', 'line', 'pcoa', '2Dpcoa'])
    parser.add_argument('-ylim', '--ylim', type=float, help="Y axis limits", nargs='*')
    #pcoa
    parser.add_argument('-env', '--env_name', type=str, help="Name of the phenotype. i.e. environment, treatment, etc.")
    parser.add_argument('-title', '--title', type=str, help="Plot title.")
    #parser.add_argument('-k', '--key', type=str, help="Key column name in metadata file.")
    #parser.add_argument('-v', '--val', type=str, help="Value column name in metadata file.")
    parser.add_argument('-m', '--meta_file', type=str, help="Metadata file for pcoa plot.")
    parser.add_argument('-cmap', '--cmap', type=str, help="Color map.", nargs='?', default='Set1')


    args = parser.parse_args()
    dataframe_file = args.file
    df = pd.read_table(dataframe_file)

    x = args.x
    y = args.y
    if args.type == 'box':
        print("box plot")
        sns.color_palette("pastel")
        sns.boxplot(x=x, y=y, hue=args.hue, data=df, palette='Set2')
    # sns.set_theme(style="ticks", palette="pastel")
    elif args.type == 'line':
        sns.lineplot(x=args.x, y=args.y, hue=args.hue, data=df)
    # sns.lineplot(x=x, y="Silhouette", hue="method", data=df, err_style="bars", ci="sd"
    if args.ylim:
        (min_y, max_y) = args.ylim
        plt.yticks(np.arange(min_y, max_y, (min_y+max_y)/10.))
        #plt.ylim(args.ylim)

    elif args.type == 'pcoa':
        df = pd.read_table(dataframe_file, header=0, index_col=0)
        print(df.head())
        sample_lst = df.columns.tolist()
        fig, dist_pc = get_pcoa(df, sample_lst, args.meta_file, args.env_name, args.title, args.cmap, plot=True)
        #fig.show()
        #fig.savefig(args.save)
    elif args.type == '2Dpcoa':
        df = pd.read_table(dataframe_file, header=0, index_col=0)
        sample_lst = df.columns.tolist()
        fig, pcoa_results = get_pcoa(df, sample_lst, args.meta_file, args.env_name, args.title, args.cmap)
        ordination = pcoa_results.samples[['PC1', 'PC2']]
        meta_df = pd.read_table(args.meta_file)
        merged_df = ordination.merge(meta_df, left_index=True, right_on="sample_name", how="left")
        print(merged_df)
        fig = sns.scatterplot(data=merged_df, x="PC1", y="PC2", hue=args.env_name)
        fig = fig.get_figure()
        fig.savefig(args.save)
        plt.show()
        return
    plt.xticks(rotation=45)
    plt.savefig(args.save)
    plt.show()

if __name__ == "__main__":
    main()