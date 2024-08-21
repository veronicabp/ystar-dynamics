from fuzzy_merge import *
import sys
sys.path.append('')
from utils import *

def merge_rightmove(hmlr_file="hmlr_for_hedonics_merge.dta", rightmove_file="rightmove_for_merge_flats.dta", output_file="rightmove_merge_keys.dta"):
	'''
	Fuzzy merge of HMLR data and Rightmove data 
	'''

	# Rightmove merge
	print("Importing HMLR data")
	hmlr_file = os.path.join(working_folder, hmlr_file)
	hmlr_data = pd.read_stata(hmlr_file)
	print("Number of rows:", len(hmlr_data.index))

	print("\nImporting Rightmove data")
	rightmove_file = os.path.join(working_folder, rightmove_file)
	rightmove_data = pd.read_stata(rightmove_file)
	print("Number of rows:", len(rightmove_data.index))

	match, _, _ = fuzzy_merge(hmlr_data, rightmove_data, pid1="property_id_x", pid2="property_id_y", to_tokenize1="address", to_tokenize2="address1", exact_ids=["property_id", "uprn"], output_vars=["property_id_x", "property_id_y", "uprn_x", "uprn_y", "merged_on"])
	match = match.rename(columns={'property_id_x':'property_id', 'property_id_y':'property_id_rm'})
	match = match[(match.uprn_x==match.uprn_y)|(match.uprn_x.isna())|(match.uprn_y.isna())]
	match['uprn'] = np.where(match['uprn_x'].notnull(), match['uprn_x'], match['uprn_y'])
	match = pick_best_match(match, 'property_id_rm', 'property_id', only_pid1=True)
	
	output_file = os.path.join(working_folder, output_file)
	match.to_stata(output_file)


def merge_zoopla():
	'''
	Fuzzy merge of HMLR data and Zoopla data 
	'''

	# Zoopla merge
	print("Importing HMLR data")
	hmlr_file = os.path.join(working_folder, f"hmlr_for_hedonics_merge.dta")
	hmlr_data = pd.read_stata(hmlr_file)
	print("Number of rows:", len(hmlr_data.index))

	print("\nImporting Zoopla data")
	zoopla_file = os.path.join(working_folder, f"zoopla_for_merge.dta")
	zoopla_data = pd.read_stata(zoopla_file)
	print("Number of rows:", len(zoopla_data.index))

	match, _, _ = fuzzy_merge(hmlr_data, zoopla_data, pid1="property_id_x", pid2="property_id_y", to_tokenize1="address", to_tokenize2="property_number", exact_ids=["property_id"], output_vars=["property_id_x", "property_id_y", "merged_on"])
	match = match.rename(columns={'property_id_x':'property_id', 'property_id_y':'property_id_zoop'})
	match = pick_best_match(match, 'property_id_zoop', 'property_id', only_pid1=True)

	output_file = os.path.join(working_folder, f"zoopla_merge_keys.dta")
	match.to_stata(output_file)


if __name__ == "__main__":
	# Merge rightmove data
	merge_rightmove()

	# Merge zoopla data
	merge_zoopla()
