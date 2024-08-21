import sys
sys.path.append('')
from utils import *

if __name__ == "__main__":
	input_folder = working_folder
	output_folder = os.path.join(working_folder, "rsi")

	# Make output directory
	os.makedirs(output_folder, exist_ok=True)

	file = os.path.join(input_folder, 'for_rent_rsi.csv')
	df = pd.read_csv(file)

	start_date=1995
	end_date=2024

	df['date'] = df['year_rm']
	df['L_date'] = df['L_year_rm']
	df = df[~df['L_date'].isna()]
	df = df[df['date']!=df['L_date']]

	# Baseline
	df_full = df.copy()


	tag = '_rent_resid'
	result = wrapper(df, start_date=start_date, end_date=end_date, price_var='d_log_rent_res', case_shiller=False, add_constant=True)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")