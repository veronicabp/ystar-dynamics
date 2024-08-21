import pandas as pd
import numpy as np
import re
from word2number import w2n
import os
import sys
from textblob import TextBlob
from dateutil.parser import parse
from math import ceil 
from spellchecker import SpellChecker
from tqdm import tqdm
from pqdm.processes import pqdm
# from pqdm.threads import pqdm
from multiprocessing import Pool, cpu_count
import time

sys.path.append('')
from utils import *
tqdm.pandas()

text_to_num = {
	'one':1,
	'two':2,
	'three':3,
	'four':4,
	'five':5,
	'six':6,
	'seven':7,
	'eight':8,
	'nine':9,
	'ten':10,
	'eleven':11,
	'twelve':12,
	'thirteen':13,
	'fourteen':14,
	'fifteen':15,
	'sixteen':16,
	'seventeen':17,
	'eighteen':18,
	'nineteen':19,
	'twenty':20,
	'thirty':30,
	'fourty':40,
	'fifty':50,
	'sixty':60,
	'seventy':70,
	'eighty':80,
	'ninety':90
}

MONTHS = [
	'january', 'february', 'march', 'april',
	'may', 'june', 'july', 'august',
	'september', 'october', 'november', 'december'
]

spell_months = SpellChecker(language=None)
spell_months.word_frequency.load_words(MONTHS)

def extract_number(text):
	'''
	identifies whether there is a number in a string
	the number may be in digits or it may be spelled out (e.g. twenty-two, 22)
	
	text : string
		string from which to extract number
	'''

	if not set(text.split()) & set(text_to_num):
		return text
	
	# Split the text into words
	words = text.split()
	# Initialize an empty list to hold the processed words
	processed_words = []
	# Initialize an empty list to hold the current spelled-out number
	spelled_number = []
	# Iterate over the words
	for word in words:
		# If the word is a digit, append it to processed_words and continue
		if word.isdigit():
			processed_words.append(word)
			continue
		# Try to convert the word to a number
		try:
			num = w2n.word_to_num(word)
			# If the conversion is successful, append the word to spelled_number
			spelled_number.append(word)
		except ValueError:
			# If the conversion fails and spelled_number is not empty, convert spelled_number to a number and append it to processed_words
			if spelled_number:
				spelled_number_str = ' '.join(spelled_number)
				try:
					num = w2n.word_to_num(spelled_number_str)
					processed_words.append(str(num))
				except ValueError as ve:
					pass
					# print(f"Cannot convert {spelled_number_str} to a number. {ve}")
				spelled_number = []
			# Append the word to processed_words
			processed_words.append(word)
	# If spelled_number is not empty after iterating over all words, convert it to a number and append it to processed_words
	if spelled_number:
		spelled_number_str = ' '.join(spelled_number)
		try:
			num = w2n.word_to_num(spelled_number_str)
			processed_words.append(str(num))
		except ValueError as ve:
			pass
			# print(f"Cannot convert {spelled_number_str} to a number. {ve}")
	# Join processed_words into a string and return it
	return ' '.join(processed_words)

def correct_months(text):
	'''
	corrects typos in month names
	
	text : string
		string to correct
	'''
	search = re.search(' .{1,12}[1-2][0-9][0-9][0-9]', text)
	if search:
		match = search.group()
		words = match.split()[:-1]
		for word in words:
			if word in MONTHS or word.isnumeric() or len(word)<=2:
				continue
			corrected = spell_months.correction(word)
			if corrected:
				text = text.replace(word, corrected)
	return text

def correct_typos(text):
	'''
	corrects typos in a string using the TextBlob function
	
	text : string
		string to clean
	'''
	# If number is attached to year, separate it 
	search = re.search('[0-9]year', text)
	if search:
		match = search.group()
		num = re.search('[0-9]', match).group()
		text = text.replace(f'{num}year', f'{num} year')

	return str(TextBlob(text).correct())

def remove_cardinal_numbers(text):
	'''
	convert cardinal numbers to regular numbers
	
	text : string
		string to clean
	'''
	for regex in ['1st', '2nd', '3rd', '[1-9]?[0-9]th']:
		search = re.search(regex, text)
		if search:
			match = search.group()
			num = int(re.search('[1-9]?[0-9]', match).group())
			text = text.replace(match, str(num))
	return text

def correct_holidays(text):
	'''
	convert holidays into months
	
	text : string
		string to clean
	'''
	holiday_months = {
		'christmas' : 'december',
		'midsummer' : 'june',
		'lady day' : 'march',
		'ladyday' : 'march',
		'michaelmas' : 'september',
		'michaelson' : 'september',
		'day' : ''
	}
	for holiday in holiday_months:
		text = text.replace(holiday, holiday_months[holiday])
	return text

def get_date(text, key_words = ['from','commencing','starting','beginning'], end_keys=['year']):
	'''
	extract a date from a string
	
	text : string
		string to from which to extract 
	key_words : list (string)
		key words that precede this type of date 
	end_keys : list (string)
		key words that follow this type of date
	'''

	regex = '('
	for word in key_words:
		regex += word + '|'
	regex = regex[:-1] + ')'

	search = re.search(f'{regex}.*', text)
	if search:
		match = search.group() + ' '

		# Go until the first year in this string
		years = re.findall('[1-4][0-9][0-9][0-9]', match)
		if len(years)>0:
			year = years[0]
			submatch = match.split(year)[0] + year

			# If we think we have other information here aside from the start date, don't risk it
			parse_date = True 
			for key in end_keys:
				if key in submatch:
					parse_date = False
					break

			if parse_date:
				# Get date
				try:
					date = parse(submatch, fuzzy=True)
					return date.strftime('%m-%d-%Y'), submatch
				except:
					# If could not get date, use year 
					# print('Could not extract year from "', submatch, '" so instead we are using "', year, '" as the date')
					return year, submatch
	return None, ''

def get_date_from(text):
	'''
	extract the lease origination date
	
	text : string
		string to clean
	'''
	return get_date(text, key_words = ['from','commencing','starting','beginning','form'], end_keys=[' to ','until','expiring','ending','terminating','year'])

def get_date_start(text):
	'''
	extract the date at the beginning of the string
	
	text : string
		string to clean
	'''
	return get_date(text, key_words = ['^'])

def get_date_to(text):
	'''
	extract the lease end date
	
	text : string
		string to clean
	'''
	return get_date(text, key_words = [' to ','until','expiring','ending','terminating','expire','end','terminate'])

def get_number_years(text, date_from=None, date_to=None):
	'''
	extract the lease length
	
	text : string
		string to clean
	date_from : string
		lease start date 
	date_to : string
		lease end date
	'''
	number_years, _ = get_number_years_exact(text)
	if number_years:
		return number_years
	search = re.search('(term|lease|for).{0,10}[1-9][0-9]?[0-9]?[0-9]?', text)
	if search:
		match = search.group()
		number_years = int(re.search('[1-9][0-9]?[0-9]?[0-9]?', match).group())
		return number_years 
	elif date_from and date_to:
		t = parse(date_to, fuzzy=True) - parse(date_from, fuzzy=True)
		return t.days/365
	return None

def get_number_years_exact(text):
	'''
	extract the lease length using a precise regex expression
	
	text : string
		string to clean
	'''
	search = re.search('[1-9][0-9]?[0-9]?[0-9]? years', text)
	if search:
		match = search.group()
		number_years = int(re.search('[1-9][0-9]?[0-9]?[0-9]?', match).group())
		return number_years, match 
	return None, ''

def add_years(text):
	'''
	flag a year in a string if it is not already flagged
	
	text : string
		string to clean
	'''
	search = re.search('^[1-9][0-9]?[0-9]?[0-9]?', text)
	if search:
		match = search.group()
		if f'{match} year' not in text:
			text = text.replace(match, f'{match} years')
	return text

def date_from_is_registration(text):
	'''
	check if the lease origination date is the registration date
	
	text : string
		string to clean
	'''
	search = re.search('((date(d)?|years) of (the |this )?(registered )?lease|commencement date|date(d)? as mentioned therein|date(d)? thereof|date(d)? hereof)', text)
	if search:
		return True 
	return False


def extract_term(original_text, date_registered=None):
	'''
	extract relevant information about a lease from the recorded text field
	
	original_text : string
		raw text recorded in the lease document
	date_registered : Date
		date that lease was registered
	'''

	if type(original_text) != str:
		return [None, None, None]

	text = original_text.lower()
	text = text.replace(',','').replace('.','-').replace('~','').replace(' year ', ' years ')
	text = " ".join(text.split())
	text = remove_cardinal_numbers(text)
	text = correct_months(text)
	text = correct_typos(text)
	text = extract_number(text)
	text = re.sub(r'less \d+ (day|week)','', text)
	text = correct_holidays(text)

	substring = text

	number_years, number_years_str = get_number_years_exact(substring)
	substring = substring.replace(number_years_str, "")

	# print("Number years:", number_years)

	# Check if date from is just the registration date
	date_from = None
	if date_from_is_registration(substring):
		date_from = date_registered

		# print("Looking for date from as date registered:", date_from)

	# If not, search for it in the text
	if date_from==None:
		date_from, date_from_str = get_date_from(substring)
		substring = substring.replace(date_from_str, "")

		# print("Looking for date from in text:", date_from)

	date_to, date_to_str = get_date_to(substring)
	substring = substring.replace(date_to_str, "")

	# If still could not find date from, check and see if it's at the beginning of the string
	if date_from==None:
		date_from, date_from_str = get_date_start(substring)
		substring = substring.replace(date_from_str, "")

		# print("Looking for date from at the beginning of the sentence:", date_from)

	if number_years == None:
		substring = add_years(substring)
		number_years = get_number_years(substring, date_from=date_from, date_to=date_to)

	if number_years == None and date_from==None and date_to==None:
		out ='\n\n\n---------------------------------------------------'
		out += "\nCould not parse the following text:"
		out += "\nText: " + original_text
		if text != original_text.lower():
			out += "\nCorrected text: " + text
		out += f"\nDate from: {date_from}"
		out += f"\nDate to: {date_to}"
		out += f"\nNumber years: {number_years}"
		# print(out)

	return [number_years, date_from, date_to]

def apply_extract_term(chunk):
	'''
	wrapper for extract_term()
	
	chunk : numpy array
		chunk of data 
	'''
	chunk[['number_years', 'date_from', 'date_to']] = chunk.progress_apply(lambda row: extract_term(row['Term'], date_registered=row['date_registered']), axis=1, result_type="expand")
	return chunk

def process_row(row):
    return extract_term(row['Term'], date_registered=row['date_registered'])

def get_merge_key(s):
	if type(s) != str:
		return s

	s = s.upper().replace('.','').replace(',','').replace("'",'').replace('(','').replace(')','').replace('FLAT','').replace('APARTMENT','')
	return " ".join(s.split())

def parallelize(df, n_jobs=5):
	df.reset_index(drop=True, inplace=True)
	rows = df.to_dict('records')

	results = pqdm(rows, process_row, n_jobs=n_jobs) 
	new_cols = pd.DataFrame(results, columns=['number_years', 'date_from', 'date_to'])
	df[['number_years', 'date_from', 'date_to']] = new_cols

	return df

def extract_lease_terms():
	# ###########################
	# # Existing lease titles
	# ###########################
	print("Extracting Open Lease Titles: (Current)")
	file = os.path.join(raw_folder, 'hmlr','LEASES_FULL_2024_02.csv')
	df = pd.read_csv(file)
	
	# Create merge key
	df['merge_key'] = df.progress_apply(lambda row: get_merge_key(row['Associated Property Description']), axis=1)
	
	# Drop if missing date or address data
	df = df[~df.merge_key.isna()]
	df = df[~df['Term'].isna()]
	df = df[['merge_key','Term','Date of Lease', 'Associated Property Description ID', 'OS UPRN', 'Unique Identifier']]

	df.loc[~df['Date of Lease'].isna(),['year_registered' ]] = df['Date of Lease'].loc[~df['Date of Lease'].isna()].astype(str).str[-4:].astype(int)
	df.loc[~df['Date of Lease'].isna(),['month_registered']] = df['Date of Lease'].loc[~df['Date of Lease'].isna()].astype(str).str[3:5].astype(int)
	df = df.rename(columns={'Date of Lease' : 'date_registered'})

	start = time.time()
	df = parallelize(df, n_jobs=n_jobs)
	end = time.time()
	print(f'Time elapsed: {round((end-start)/60,2)} minutes.')

	output_file = os.path.join(working_folder, 'extracted_terms_open.csv')
	df.to_csv(output_file, index=False)

	###########################
	# Historical lease titles
	###########################
	print("Extracting Open Lease Titles (May 2023):")
	file = os.path.join(raw_folder, 'hmlr','LEASES_FULL_2023_05.csv')
	df = pd.read_csv(file)

	# Create merge key
	df['merge_key'] = df.progress_apply(lambda row: get_merge_key(row['Associated Property Description']), axis=1)
	
	# Drop if missing date or address data
	df = df[~df.merge_key.isna()]
	df = df[~df['Term'].isna()]
	df = df[['merge_key','Term','Date of Lease', 'Associated Property Description ID', 'OS UPRN', 'Unique Identifier']]

	df.loc[~df['Date of Lease'].isna(),['year_registered' ]] = df['Date of Lease'].loc[~df['Date of Lease'].isna()].astype(str).str[-4:].astype(int)
	df.loc[~df['Date of Lease'].isna(),['month_registered']] = df['Date of Lease'].loc[~df['Date of Lease'].isna()].astype(str).str[3:5].astype(int)
	df = df.rename(columns={'Date of Lease' : 'date_registered'})

	start = time.time()
	df = parallelize(df, n_jobs=n_jobs)
	end = time.time()
	print(f'Time elapsed: {round((end-start)/60,2)} minutes.')

	output_file = os.path.join(working_folder, 'extracted_terms_open_may2023.csv')
	df.to_csv(output_file, index=False)

	################ June:
	# print("Extracting Open Lease Titles (June 2022):")
	# file = os.path.join(raw_folder, 'hmlr','LEASES_FULL_2022_06.csv')
	# df = pd.read_csv(file)

	# # Create merge key
	# df['merge_key'] = df.progress_apply(lambda row: get_merge_key(row['Associated Property Description']), axis=1)
	
	# # Drop if missing date or address data
	# df = df[~df.merge_key.isna()]
	# df = df[~df['Term'].isna()]
	# df = df[['merge_key','Term','Date of Lease', 'Associated Property Description ID', 'OS UPRN', 'Unique Identifier']]

	# df.loc[~df['Date of Lease'].isna(),['year_registered' ]] = df['Date of Lease'].loc[~df['Date of Lease'].isna()].astype(str).str[-4:].astype(int)
	# df.loc[~df['Date of Lease'].isna(),['month_registered']] = df['Date of Lease'].loc[~df['Date of Lease'].isna()].astype(str).str[3:5].astype(int)
	# df = df.rename(columns={'Date of Lease' : 'date_registered'})

	# start = time.time()
	# df = parallelize(df, n_jobs=n_jobs)
	# end = time.time()
	# print(f'Time elapsed: {round((end-start)/60,2)} minutes.')

	# output_file = os.path.join(working_folder, 'extracted_terms_open_june2022.csv')
	# df.to_csv(output_file, index=False)

	###########################
	# Closed lease titles
	###########################
	print("Extracting Closed Lease Titles:")

	dfs = []
	folder = os.path.join(raw_folder, "hmlr", "purchased_titles")
	files = [file for file in os.listdir(folder) if file.endswith('xlsx')]
	for file in files:
		print(file)
		file = os.path.join(folder, file)
		this_df = pd.read_excel(file)
		dfs.append(this_df)
	df = pd.concat(dfs, ignore_index=True)
	df = df.rename(columns={'TITLE_CLOS_DATE' : 'date_registered', 'TERM':'Term', 'CLOS_YEAR':'year_registered'})
	df['Term'] = df['Term'].str.replace("_x000D_", " ")
	df['Term'] = df['Term'].replace(r'\s+|\\n', ' ', regex=True) 

	start = time.time()
	df = parallelize(df, n_jobs=n_jobs)
	end = time.time()
	print(f'Time elapsed: {round((end-start)/60,2)} minutes.')
	
	output_file = os.path.join(working_folder, 'extracted_terms_closed.csv')
	df.to_csv(output_file, index=False)

	###########################
	# Other purchased titles
	###########################
	file = os.path.join(raw_folder, "hmlr", "purchased_titles", "Princeton Lease Information for Issue.csv")
	df = pd.read_csv(file, encoding='latin1')
	df = df.rename(columns={'Date Registered' : 'date_registered', 'Registered Lease Details':'Term', 'Client Reference':'transaction_id'})
	df['date_registered'] = pd.to_datetime(df['date_registered'], format='mixed')
	df['date_registered'] = df['date_registered'].dt.strftime('%d/%m/%Y')

	start = time.time()
	df = parallelize(df, n_jobs=n_jobs)
	end = time.time()
	print(f'Time elapsed: {round((end-start)/60,2)} minutes.')
	
	output_file = os.path.join(working_folder, 'extracted_terms_other.csv')
	df.to_csv(output_file, index=False)

if __name__=="__main__":
	extract_lease_terms()

