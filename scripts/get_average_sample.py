import sys
sys.path.append('./L2-UniFrac')
sys.path.append('./L2-UniFrac/src')
import L2Unifrac as L2U
import argparse

def main():
    parser = argparse.ArgumentParser(description="Takes in a group of samples in an otu-table-liked and returns the average sample of these samples.")
    parser.add_argument('-i', '--input_file', type=str, help="A file with row names being OTU/taxid and column names being sample IDs.")
    parser.add_argument('-o', '--output_file', type=str, help="Ouput file name.")
    parser.add_argument('-t', '--type', type=str, choices=['wgs', '16s'], help="Data type. Accepts: wgs, 16s.")


