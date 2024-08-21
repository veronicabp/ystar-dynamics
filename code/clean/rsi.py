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
	# df = df[df.area=='B']

	start_date=data_start_year*4 + data_start_month
	end_date=data_end_year*4 + data_end_month

	df['date'] = df['year']*4 + df['quarter']
	df['L_date'] = df['L_year']*4 + df['L_quarter']
	df = df[df['date']!=df['L_date']]
	df_inc_flip = df.copy()
	df = df[df.years_held>=2]

	###################################
	# Get weights
	###################################
	outfile=f"rsi_residuals.csv"
	residuals = wrapper(df, start_date=start_date, end_date=end_date, func=apply_rsi_residuals)
	residuals.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")
	# residuals = pd.read_csv(os.path.join(output_folder, outfile))

	X = residuals[['years_held','distance']]
	X = sm.add_constant(X)
	y = residuals['residuals']**2
	result = sm.OLS(y,X).fit()
	params = result.params

	df['b_cons'] = result.params[0]
	df['b_years_held'] = result.params[1]
	df['b_distance'] = result.params[2]

	df_inc_flip['b_cons'] = result.params[0]
	df_inc_flip['b_years_held'] = result.params[1]
	df_inc_flip['b_distance'] = result.params[2]

	###################################
	# Including flippers
	###################################
	tag = '_flip'
	result = wrapper(df_inc_flip, start_date=start_date, end_date=end_date)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")

	###################################
	# Main
	###################################
	print('Main:')
	metadata = {
		'':'d_log_price',
		'_linear':'d_pres_linear'
	}

	for i, tag in enumerate(metadata):
		result = wrapper(df, start_date=start_date, end_date=end_date, price_var=metadata[tag])
		outfile=f"rsi{tag}.csv"
		result.to_csv(os.path.join(output_folder, outfile), index=False)
		print(f"Saved to {outfile}:")

	##################################
	# No weights
	##################################
	tag = '_bmn'
	result = wrapper(df, start_date=start_date, end_date=end_date, case_shiller=False)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")

	##################################
	# No constant
	##################################

	tag = '_nocons'
	result = wrapper(df, start_date=start_date, end_date=end_date, case_shiller=False, add_constant=False)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")

	###################################
	# Full
	###################################

	# Get all dates for each property 
	all_dates = df.groupby('property_id')['date'].agg(lambda x: set(x)).reset_index(name='dates_to_connect')
	all_L_dates = df.groupby('property_id')['L_date'].agg(lambda x: set(x)).reset_index(name='L_dates_to_connect')
	df = df.merge(all_dates, on='property_id')
	df = df.merge(all_L_dates, on='property_id')
	df['dates_to_connect'] = df.apply(lambda row: sorted(row.dates_to_connect.union(row.L_dates_to_connect)), axis=1)

	print(df[df.has_extension==1][['property_id','date','L_date','dates_to_connect']])

	# tag = '_full_connect_all'
	# result = wrapper(df, start_date=start_date, end_date=end_date, func=apply_rsi_full, connect_all=True)
	# outfile=f"rsi{tag}.csv"
	# result.to_csv(os.path.join(output_folder, outfile), index=False)
	# print(f"Saved to {outfile}:")

	tag = '_full'
	result = wrapper(df, start_date=start_date, end_date=end_date, func=apply_rsi_full, connect_all=False)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")

	###################################
	# Yearly
	###################################
	df['date'] = df['year']
	df['L_date'] = df['L_year']
	df = df[df['date']!=df['L_date']]

	start_date = data_start_year 
	end_date = data_end_year

	tag = '_yearly'
	result = wrapper(df, start_date=start_date, end_date=end_date)
	outfile=f"rsi{tag}.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")
