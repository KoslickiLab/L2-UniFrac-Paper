import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Get plot from a dataframe.")
    parser.add_argument('-f', '--file', type=str, help="Dataframe file.")
    parser.add_argument('-x', '--x', type=str, help="x axis.")
    parser.add_argument('-y', '--y', type=str, help="y axis.")
    parser.add_argument('-s', '--save', type=str, help="If wants to save df for future use, file name to save as")
    parser.add_argument('-p', '--phenotype', type=str, help="Condition/site of interest.")
    parser.add_argument('-c', '--column', type=str, help="Column name in which -p input falls.")
    parser.add_argument('-h', '--hue', type=str, help="Hue", nargs='?', default="Method")
    parser.add_argument('-t', '--type', type=str, help="Plot type.", choices=['box', 'line'])

    args = parser.parse_args()
    dataframe_file = args.file
    df = pd.read_table(dataframe_file, index_col=0)
    s = pd.Series(list(df[args.column]))

    try:
        args.phenotype in s.values
    except:
        print("Phenotype not found in column specified.")

    if args.column:
        df = df[df[args.column] == args.phenotype]

    if args.type == 'box':
        sns.boxplot(x=args.x, y=args.y, hue=args.hue)
    # sns.set_theme(style="ticks", palette="pastel")
    elif args.type == 'line':
        sns.lineplot(x=args.x, y=args.y, hue=args.hue, data=df)
    # sns.lineplot(x=x, y="Silhouette", hue="method", data=df, err_style="bars", ci="sd")
    plt.savefig(args.save)

