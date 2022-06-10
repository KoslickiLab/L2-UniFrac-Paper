import sys
sys.path.append('../L2Unifrac')
sys.path.append('../L2Unifrac/src')
sys.path.append('../src')
import L2Unifrac as L2U

def test_push_up_from_wgs_profile():
    print(L2U.push_up_from_wgs_profile('wgs-env0-sample24-reads.profile'))

def test_merge_efficiency():
    return


if __name__ == '__main__':
	test_push_up_from_wgs_profile()