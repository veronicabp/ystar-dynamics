import sys
sys.path.append('')
from utils import *

if __name__ == "__main__":
	input_folder = working_folder
	output_folder = os.path.join(working_folder, "rsi")

	# Make output directory
	os.makedirs(output_folder, exist_ok=True)

	file = os.path.join(input_folder, 'for_rsi.csv')
	df = pd.read_csv(file)

	start_date=data_start_year*4 + data_start_month
	end_date=data_end_year*4 + data_end_month

	df['date'] = df['year']*4 + df['quarter']
	df['L_date'] = df['L_year']*4 + df['L_quarter']
	df = df[df['date']!=df['L_date']]
	df_inc_flip = df.copy()
	df = df[df.years_held>=2]

	# Create placebo extensions 
	df = df[df.extension==0]
	random_values = np.random.choice([0, 1], size=len(df), p=[0.99,0.01])
	df['extension'] = random_values

	print(df[['property_id','d_log_price','whb_duration','extension']])
	print('Num ext:', len(df[df.extension==1]))
	print('Num control:', len(df[df.extension==0]))

	tag = '_placebo_wcons'
	result = wrapper(df, start_date=start_date, end_date=end_date, case_shiller=False)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")

	##################################
	# No constant
	##################################

	tag = '_placebo_nocons'
	result = wrapper(df, start_date=start_date, end_date=end_date, case_shiller=False, add_constant=False)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")


