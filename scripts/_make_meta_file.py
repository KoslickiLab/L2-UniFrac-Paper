import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="This helper script creates metadata file to be used for pcoa plot"
                                                 "given a pairwise distance matrix file.")
    parser.add_argument('-f', '--file', type=str, help="Pairwise distance file.")
    parser.add_argument('-o', '--output', type=str, help="Output file to write to.")
    parser.add_argument('-v', '--env_name', type=str, help="Environment name.e.g. body site, treatment")

    args = parser.parse_args()

    df = pd.read_table(args.file, header=0, index_col=0)
    meta_df = pd.DataFrame(columns=args.env_name, index=df.columns) #col names = environment name, sample name = phenotypes
    meta_df[args.env_name] = df.columns
    meta_df.to_csv(args.output, sep='\t')


if __name__ == "__main__":
    main()