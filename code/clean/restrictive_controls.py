import sys
sys.path.append('')
from utils import *

def restrictive_controls(row, control_dict, price_var='d_log_price', restrict_price=False):
	# text = f"\n\n{'='*100}\nFinding controls for {row['property_id']} in dates {row['date']} and {row['L_date']} with duration in 2023 of {row['duration2023']}.\n\n"

	# Restrict by duration
	controls = control_dict[row.duration2023]

	# Restrict by price 
	if restrict_price:
		controls = controls[abs(controls['L_log_price'] - row.L_log_price)<=0.1]

	if len(controls)==0:
		return [None]*5

	# Restrict by distance
	controls['distance'] = controls.apply(lambda x: haversine(row.lat_rad, row.lon_rad, x['lat_rad'], x['lon_rad']), axis=1)
	radius = np.ceil(controls['distance'].min())
	controls = controls[controls['distance']<=radius]

	# text += f"\n\nFINAL RADIUS: {radius}\n\n"
	# text += f"Restricted Controls:\n\n\n{controls[['property_id', 'date', 'L_date', 'duration2023', 'distance', 'log_price','L_log_price','d_log_price']]}\n\n"
	# print(text)

	return [controls.log_price.mean(), controls.L_log_price.mean(), controls.d_log_price.mean(), len(controls), radius]

def apply_restrictive_controls(extensions, control_dict, price_var='d_log_price', case_shiller=False, connect_all=False, add_constant=False):
	extensions[['log_price_ctrl','L_log_price_ctrl','d_log_price_ctrl','num_controls','radius']] = extensions.apply(lambda row: restrictive_controls(row, control_dict, price_var=price_var), axis=1, result_type="expand")
	return extensions

if __name__ == "__main__":
	input_folder = working_folder
	output_folder = os.path.join(working_folder, "rsi")

	# Make output directory
	os.makedirs(output_folder, exist_ok=True)

	file = os.path.join(input_folder, 'for_rsi.csv')
	df = pd.read_csv(file)

	start_date=1995
	end_date=2023
	df['date'] = df['year']
	df['L_date'] = df['L_year']

	df = df[df['date']!=df['L_date']]
	df = df[df['years_held']>2]

	result = wrapper(df, start_date=start_date, end_date=end_date, func=apply_restrictive_controls, groupby=['area','date','L_date'])
	outfile=f"restrictive_controls.csv"
	result.to_csv(os.path.join(output_folder, outfile), index=False)
	print(f"Saved to {outfile}:")
