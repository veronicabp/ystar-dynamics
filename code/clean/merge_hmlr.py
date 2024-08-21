
import sys
sys.path.append('')
from utils import *
from fuzzy_merge import *

def merge_hmlr():
	'''
	Fuzzy merge of HMLR data 
	'''
	# HMLR merge
	print("Importing price data")
	transaction_file = os.path.join(working_folder, "price_data_for_merge.csv")
	transaction_data = pd.read_csv(transaction_file)
	print("Number of rows:", len(transaction_data.index))

	print("\nImporting lease data")
	lease_file = os.path.join(working_folder, "lease_data_for_merge.csv")
	lease_data = pd.read_csv(lease_file)
	print("Number of rows:", len(lease_data.index))

	pid1 = 'property_id'
	pid2 = 'merge_key'
	output_file = os.path.join(working_folder, "hmlr_merge_keys.dta")
	match, _, _ = fuzzy_merge(transaction_data, lease_data, pid1=pid1, pid2=pid2, to_tokenize1="address", to_tokenize2="address", exact_ids=["merge_key_1", "merge_key_2"], output_vars=['property_id','merge_key','merged_on'])

	############################
	# Drop duplicates matches
	############################
	match = pick_best_match(match, pid1, pid2)

	# Export 
	match.to_stata(output_file, write_index=False)

if __name__=="__main__":
	merge_hmlr()
