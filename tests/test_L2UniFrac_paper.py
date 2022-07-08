import sys
sys.path.append('L2-UniFrac')
sys.path.append('L2-UniFrac/src')
sys.path.append('src')
sys.path.append('scripts')
import L2UniFrac as L2U
import partition_predict as pp

def test_push_up_from_wgs_profile():
    print(L2U.push_up_from_wgs_profile('wgs-env0-sample24-reads.profile'))

def test_merge_efficiency():
    return

def test_merge_profiles_by_dir():
    L2U.merge_profiles_by_dir('profile_test')

def test_partition():
    group_samples = pp.extract_samples_by_group(pp.biom_file, pp.metadata_file, pp.metadata_key)
    print(group_samples)

if __name__ == '__main__':
	test_partition()