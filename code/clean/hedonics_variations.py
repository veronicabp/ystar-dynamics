import sys
sys.path.append('')
from utils import *

def get_rsi(inp):
	area, duration, sub_ad, dates, hedonics_vars, tags = inp 
	data = {
		'area':[area]*len(dates),
		'duration2023_10yr':[duration]*len(dates),
		'date':dates
	}

	for i, hedonic in enumerate(hedonics_vars):

		tag = tags[i]
		var = f'rsi_{tag}'
		data[var] = []

		sub = sub_ad[~sub_ad[hedonic].isna()]

		if len(sub)<=10:
			biggest_group = []
			params = []

		else:
			# Select largest connected group 
			uf = get_union(sub)
			connected_groups = uf.build_groups()
			biggest_group = []
			for key in connected_groups:
				if len(connected_groups[key])>len(biggest_group):
					biggest_group = sorted(connected_groups[key])
			
			params, _, summary = rsi(sub, price_var=hedonic, dummy_vars=[f'd_{date}' for i, date in enumerate(biggest_group) if i!=0], add_constant=False)

		for i, date in enumerate(dates):
			if date not in biggest_group:
				param = None 
			else:
				idx = biggest_group.index(date)
				param = params[idx]

			data[var].append(param)

	return pd.DataFrame(data)

if __name__=="__main__":
	file = os.path.join(working_folder, 'for_rsi_large.csv')
	df = pd.read_csv(file)

	start_date=data_start_year*4 + data_start_month
	end_date=data_end_year*4 + data_end_month

	df['date'] = df['year']*4 + df['quarter']
	df['L_date'] = df['L_year']*4 + df['L_quarter']
	df = df[df['date']!=df['L_date']]
	df['duration2023_10yr'] = np.round(df['duration2023']/10)*10

	durations = sorted(df['duration2023_10yr'].unique())
	areas = sorted(df['area'].unique())
	hedonics_vars = [var for var in df.columns if var.startswith('d_pres')] + [var for var in df.columns if var.startswith('d_tpres')]
	tags = hedonics_vars

	# Remove extensions 
	extensions = df[df.extension==1]
	controls = df[df.extension==0]
	controls = get_dummies(controls, start_date=start_date, end_date=end_date)

	# Parallelize 
	pool = Pool(n_jobs)

	inp = []
	max_len = 0
	for area in areas:
		sub_a = controls[(controls['area']==area)]
		for duration in durations:
			sub_ad = sub_a[(sub_a['duration2023_10yr']==duration)]
			dates = sorted(set(sub_ad['date'].unique()) | set(sub_ad['L_date'].unique()))
			if len(sub_ad)>=10:
				inp.append((area,duration, sub_ad, dates, hedonics_vars, tags))
			if len(sub_ad)>max_len:
				max_len = len(sub_ad)

	print("Max length of a df:", max_len)
	print("Number of groups:", len(inp))

	results = pqdm(inp, get_rsi, n_jobs=n_jobs) 
	valid_results = [r for r in results if isinstance(r, (pd.DataFrame, pd.Series))]
	invalid_results = [r for r in results if not isinstance(r, (pd.DataFrame, pd.Series))]
	if len(invalid_results)>0:
		print(f"Error! {invalid_results}")
	output = pd.concat(valid_results)

	file = os.path.join(working_folder, 'rsi_variations.csv')
	output.to_csv(file, index=False)
		
