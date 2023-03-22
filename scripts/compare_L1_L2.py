import argparse
import matplotlib.pyplot as plt
import os
import sys

# try:
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#     sys.path.append(os.path.dirname(SCRIPT_DIR))
# except:
#     pass

sys.path.append("src")
from helper import get_scatter_plot_from_2_dist_matrices

def main():
    parser = argparse.ArgumentParser(description="Get plot comparing pairwise L1UniFrac and L2UniFrac")
    parser.add_argument("-f1", "--file1", type=str, default="data/714_mouse/pairwise_L1Unifrac_all.tsv")
    parser.add_argument("-f2", "--file2", type=str, default="data/714_mouse/pairwise_L2UniFrac_all.tsv")
    parser.add_argument("-s", "--save", type=str, default="data/714_mouse/L1_L2_correlation.png")
    args = parser.parse_args()
    get_scatter_plot_from_2_dist_matrices(args.file1, args.file2, args.save)


if __name__ == "__main__":
    main()
