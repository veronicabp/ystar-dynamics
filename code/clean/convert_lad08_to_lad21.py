import sys
sys.path.append('')
from utils import *

if __name__ == "__main__":
	ons_folder = os.path.join(raw_folder, "ons")

	lad08_file = os.path.join(ons_folder, 'LAD_DEC_2008_GB_BFC', 'LAD_DEC_2008_GB_BFC.shp')
	lad21_file = os.path.join(ons_folder, 'LAD_DEC_2021_UK_BFC', 'LAD_DEC_2021_UK_BFC.shp')
	hilber_file = os.path.join(raw_folder,"original","ecoj12213-sup-0001-DataS1","dta files","data LPA.dta")

	lad08_gdf = gpd.read_file(lad08_file)
	lad21_gdf = gpd.read_file(lad21_file)
	hilber = pd.read_stata(hilber_file)

	fields = ['pdevel90_m2','refusal_maj_7908', 'delchange_maj1', 'delchange_maj5', 'delchange_maj6', 'rindex2', 'male_earn_real']
	hilber = hilber[['lpa_code','lpa_name', 'year']+fields]
	
	# Calculate total area of each codes system
	lad08_gdf['lad08_area'] = lad08_gdf.area
	lad21_gdf['lad21_area'] = lad21_gdf.area

	# Calculate overlap
	# Spatial join - this associates each LPA with an LAD
	print('Calculating intersection:')
	join_gdf = gpd.sjoin(lad21_gdf, lad08_gdf, how='inner', predicate='intersects')

	# Calculate intersection area
	join_gdf['intersection_area'] = join_gdf.apply(
		lambda row: row['geometry'].intersection(
			lad08_gdf.loc[lad08_gdf['LAD08CD'] == row['LAD08CD'], 'geometry'].values[0]
		).area, axis=1
	)

	# Merge in Hilber data to 2008 data
	join_gdf = join_gdf.merge(hilber, left_on='LAD08CD', right_on='lpa_code', how='left')
	print('Joined GDF:\n', join_gdf)

	# Aggregate by 2021 codes, taking weighted mean 
	for field in fields:
		join_gdf[field] = join_gdf[field]*join_gdf['intersection_area']/join_gdf['lad21_area']

	# Collapse
	agg = {field:'sum' for field in fields}
	agg['lad21_area']='first'
	df = join_gdf.groupby(['LAD21NM','LAD21CD', 'year']).agg(agg).reset_index()
	df = df[['LAD21NM','LAD21CD','year']+fields]
	df = df[df.LAD21CD.str.startswith('E')]

	# Drop Isles of Scilly (missing data)
	df = df[df.LAD21CD!='E06000053']

	df = df.rename(columns={'LAD21CD':'lpa_code', 'LAD21NM':'lpa_name'})

	output_file = os.path.join(clean_folder,'hilber_lad21.dta')
	df.to_stata(output_file, write_index=False)
	print(f'Exported {output_file}.')
