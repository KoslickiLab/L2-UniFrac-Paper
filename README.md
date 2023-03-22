# L2-UniFrac-Paper
### Installation

```
git clone https://github.com/KoslickiLab/L2-UniFrac-Paper.git
cd L2-UniFrac-Paper
conda create -n L2UniFrac python=3.10
conda activate L2UniFrac
python -m pip install -r requirements.txt
git clone https://github.com/KoslickiLab/L2-UniFrac.git
```
### Download data
```angular2html
wget https://zenodo.org/record/7563168/files/L1-UniFrac-Out.csv -o data/L1-UniFrac-Out.csv
wget https://zenodo.org/record/7566079/files/L2-UniFrac-Out.csv -o data/L2-UniFrac-Out.csv

```

### 1. L2UniFrac clustering comparison
```
bash reproduce_clustering.sh
```
### 2. Finding representative samples
```
bash reproduce_representative_pcoa.sh
```
### 3. Classification
```
bash reproduce_classification_16s_body_sites.sh
```
### 4. Differential abundance analysis
```angular2html
python scripts/get_wgs_diffabund.py -m data/hmgdb_adenoma_bioproject266076.csv -d data/adenoma_266076/profiles -s data/adenoma_266076 -t phylum -prefix adenoma_phylum
python scripts/get_wgs_diffabund.py -m data/hmgdb_adenoma_bioproject266076.csv -d data/adenoma_266076/profiles -s data/adenoma_266076 -t phylum -prefix adenoma_genus

```
### 5. Compare L1 and L2 UniFrac
```angular2html
python scripts/compare_L1_L1.py
```