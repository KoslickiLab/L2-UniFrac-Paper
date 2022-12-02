from ete3 import NCBITaxa
ncbi = NCBITaxa()
#ncbi.update_taxonomy_database()
import pandas as pd
import argparse


#' Desired format
#' taxon	total	A	B	C
#' "k__Bacteria";"p__Actinobacteria";"c__Actinobacteria";...	1	0	1	0
#' "k__Bacteria";"p__Actinobacteria";"c__Actinobacteria";...	1	0	1	0
#' "k__Bacteria";"p__Actinobacteria";"c__Actinobacteria";...	1	0	1	0
#'

def parse_detail(abundance_file, save_as):
    df = pd.read_table(abundance_file, header=0)
    print(df)
    col_names = ['otu_id', 'lineage']
    col_names.extend(df.columns[1:].tolist()) #not including the first column taxid
    combined_df = pd.DataFrame(columns=col_names)
    taxon_col = []
    otu_id_col = []
    for taxid in df['taxid']:
        if taxid == -1 or 'species' not in ncbi.get_rank([taxid]).values():
            df.drop(df[df['taxid'] == taxid].index, axis=0, inplace=True) #remove this row
        else:
            otu_id_col.append(str(taxid))
            lineage = ncbi.get_lineage(taxid)
            name_path = get_name_path(lineage)
            # convert lineage to scientific names
            # concatenate values in lineage_translator with ';' in the order of lineage
            taxon_col.append(name_path)
    for environment in df.columns[1:]:
        combined_df[environment] = df[environment]
    combined_df['lineage'] = taxon_col
    combined_df['otu_id'] = otu_id_col
    combined_df.to_csv(save_as, sep='\t', index=False)
    print(combined_df)


def get_name_path(lineage):
    valid_ranks = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus','species']
    ranks = ncbi.get_rank(lineage)
    lineage_translator = ncbi.get_taxid_translator(lineage)
    name_path_list = []
    for taxid in lineage[2:]:
        if ranks[taxid] in valid_ranks:
            if ranks[taxid] == 'superkingdom':
                r = 'k'
            else:
                r = ranks[taxid][0]
            name = r + '__' + lineage_translator[taxid]
            name_path_list.append(name)
    name_path = ';'.join(name_path_list)
    return name_path

def main():
    parser = argparse.ArgumentParser(description='Get representative samples from a metadata file.')
    parser.add_argument('-s', '--save', type=str, help="Save output file as.")
    parser.add_argument('-i', '--input_file', type=str, help="Input file. An otu-like file that contains the abundances "
                                                             "of representative samples at corresponding taxid.")
    args = parser.parse_args()
    parse_detail(args.input_file, args.save)
    return


if __name__ == '__main__':
    main()