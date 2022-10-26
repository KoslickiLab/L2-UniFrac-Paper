import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('src')
sys.path.append('scripts')
import numpy as np
from parse_data import parse_otu_table, parse_otu_table_no_extend
import biom
import extract_data as extract



#toy sample
toy_data = np.arange(40).reshape(10, 4)
toy_sample_ids = ['S%d' % i for i in range(4)]
toy_observ_ids = ['O%d' % i for i in range(10)]
toy_sample_metadata = [{'environment': 'A'}, {'environment': 'B'},
                   {'environment': 'A'}, {'environment': 'B'}]
toy_observ_metadata = [{'taxonomy': ['Bacteria', 'Firmicutes']},
                   {'taxonomy': ['Bacteria', 'Firmicutes']},
                   {'taxonomy': ['Bacteria', 'Proteobacteria']},
                   {'taxonomy': ['Bacteria', 'Proteobacteria']},
                   {'taxonomy': ['Bacteria', 'Proteobacteria']},
                   {'taxonomy': ['Bacteria', 'Bacteroidetes']},
                   {'taxonomy': ['Bacteria', 'Bacteroidetes']},
                   {'taxonomy': ['Bacteria', 'Firmicutes']},
                   {'taxonomy': ['Bacteria', 'Firmicutes']},
                   {'taxonomy': ['Bacteria', 'Firmicutes']}]
toy_table = biom.Table(toy_data, toy_observ_ids, toy_sample_ids, toy_observ_metadata, toy_sample_metadata, table_id='Example Table')

def test_otu_present():
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    _, _, nodes_in_order = extract.parse_tree_file(tree_file)
    sample_vector_dict, sample_ids, otus = parse_otu_table_no_extend('data/714_mouse/otu_table.tsv')
    assert set(otus).issubset(set(nodes_in_order))

def test_parse_otu_table_no_extend():
    otu_file = 'data/1928_body_sites/47422_otu_table.tsv'
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    _, _, nodes_in_order = extract.parse_tree_file(tree_file)
    sample_vector_dict, sample_ids, otus = parse_otu_table_no_extend(otu_file, normalize=True)
    assert len(sample_vector_dict["1928.SRS015139.SRX020561.SRR043887"]) == len(nodes_in_order)
    assert np.isclose(np.sum(sample_vector_dict["1928.SRS015139.SRX020561.SRR043887"]), 1)

def test_extract_samples_direct():
    biom_file = 'data/1928_body_sites/47422_otu_table.biom'
    tree_file = 'data/trees/gg_13_5_otus_99_annotated.tree'
    nodes_weighted, sample_ids = extract.extract_samples_direct(biom_file, tree_file)
    print(len(nodes_weighted['1928.SRS048971.SRX020527.SRR047153']))

test_extract_samples_direct()