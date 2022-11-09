import argparse
import sys
sys.path.append('./src')
from helper import convert_profiles_to_otu


def argument_parser():
    parser = argparse.ArgumentParser(description="This function takes in a directory as an input. For any file in the "
                                                 "directory, if it ends with .profile, convert it to an otu like file"
                                                 "and save it in the output directory.")
    parser.add_argument('-i', '--in_dir', type=str, required=True, help='Path to the directory containing profiles to'
                                                                        'be converted')
    parser.add_argument('-o', '--output_file', type=str, help='File path to save the new otu file as.')
    return parser

def main():
    parser = argument_parser()
    args = parser.parse_args()
    convert_profiles_to_otu(args.in_dir, args.output_file)


if __name__ == "__main__":
    main()