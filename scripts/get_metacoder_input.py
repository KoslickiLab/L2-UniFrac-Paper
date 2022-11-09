from ete3 import NCBITaxa
ncbi = NCBITaxa()
#ncbi.update_taxonomy_database()
import pandas as pd
import os


#' Desired format
#' taxon	total	A	B	C
#' "k__Bacteria";"p__Actinobacteria";"c__Actinobacteria";...	1	0	1	0
#' "k__Bacteria";"p__Actinobacteria";"c__Actinobacteria";...	1	0	1	0
#' "k__Bacteria";"p__Actinobacteria";"c__Actinobacteria";...	1	0	1	0
#'

def parse(abundance_files, save_as):
    '''

    :param abundance_files: List of abundance_files. assumes all the files have the same ID column in the same order.
    Column names are 'ID' and 'relative_abundance'. Case sensitive for now
    :return:
    '''
    #initiate dataframe with columns being files in abundance_files
    abundance_file_names = [os.path.split(abundance_file)[-1] for abundance_file in abundance_files]
    col_names = ['taxon'] + ['total'] + abundance_file_names
    df = pd.read_table(abundance_files[0], header=0)
    row_col = df['ID'].tolist()
    row_col.remove(-1)
    combined_df = pd.DataFrame(columns=col_names, index=row_col)
    #get the 'taxon' column
    taxon_col = []
    for taxid in df['ID']:
        if taxid == -1: #remove first
            continue
        lineage = ncbi.get_lineage(taxid)
        name_path = get_name_path(lineage)
        # convert lineage to scientific names
        # concatenate values in lineage_translator with ';' in the order of lineage
        taxon_col.append(name_path)
    combined_df['taxon'] = taxon_col
    for abundance_file in abundance_files:
        #read abundance file into dataframe and get ID column
        df = pd.read_table(abundance_file, header=0)
        rel_abund = df['relative_abundance'].tolist()
        file_name = os.path.split(abundance_file)[-1]
        combined_df[file_name] = rel_abund[:-1] #exclude the last one -1 for now
    sample_df = combined_df[abundance_file_names]
    combined_df['total'] = sample_df.sum(axis=1)
    combined_df.to_csv(save_as, sep='\t', index=False)

def parse_detail(abundance_files, save_as):
    abundance_file_names = [os.path.split(abundance_file)[-1] for abundance_file in abundance_files]
    col_names = ['otu_id'] + ['lineage'] + abundance_file_names
    df = pd.read_table(abundance_files[0], header=0)
    id_col = df['ID'].tolist()
    id_col.remove(-1)
    combined_df = pd.DataFrame(columns=col_names)
    combined_df['otu_id'] = id_col
    taxon_col = []
    for taxid in df['ID']:
        if taxid == -1:  # remove first
            continue
        lineage = ncbi.get_lineage(taxid)
        name_path = get_name_path(lineage)
        # convert lineage to scientific names
        # concatenate values in lineage_translator with ';' in the order of lineage
        taxon_col.append(name_path)
    combined_df['lineage'] = taxon_col
    for abundance_file in abundance_files:
        # read abundance file into dataframe and get ID column
        df = pd.read_table(abundance_file, header=0)
        rel_abund = df['relative_abundance'].tolist()
        file_name = os.path.split(abundance_file)[-1]
        combined_df[file_name] = list(map(lambda x:x*1000, rel_abund[:-1])) # exclude the last one -1 for now
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

def write_sample_file():
    df = pd.DataFrame(columns=['sample_id','condition'])
    df['sample_id'] = ['cancer.txt','control.txt']
    df['condition'] = ['cancer','control']
    print(df)
    df.to_csv('data/adenoma_cancer_control_sample_metacoder.txt', sep='\t', index=False)


#get_name_path([1, 131567, 2, 1783272, 1239, 186801, 186802, 216572, 216851, 853])
parse_detail(['data/cancer.txt', 'data/control.txt'], 'data/adenoma_cancer_control_metacoder_input.txt')
#write_sample_file()