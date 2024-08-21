import sys
sys.path.append('')
from utils import *
from openai import OpenAI


# Main Wrapper to Pass Training files to LLM
def main():

	client = OpenAI()
	  
	assistant = client.beta.assistants.create(
	  name="Renovations",
	  instructions="Determine whether a property description implies that the property was recently renovated or improved. Respond only Yes or No.",
	  model="gpt-3.5-turbo",
	)
	thread = client.beta.threads.create()

	file = os.path.join(working_folder, 'rightmove_descriptions_matched_subset.dta')
	df = pd.read_stata(file)
	print(df)

	# Create ID for each prompt, and store results in case of duplicates
	df['summary_id'] = df['summary'].astype('category').cat.codes
	id_to_summary = dict(enumerate(df['summary'].astype('category').cat.categories))
	processed = dict()

	if not 'renovated' in df.columns:
		df['renovated'] = ''

	for i, row in tqdm(df.iterrows(), total=len(df)):

		if row['renovated']!='':
			continue

		if row['summary_id'] in processed:
			df.loc[df.index==i, 'renovated'] = processed[row['summary_id']]

		my_prompt = row['summary']
		print(my_prompt, '\n\n\n')

		message = client.beta.threads.messages.create(
			thread_id=thread.id,
			role="user",
			content=my_prompt
		)

		run = client.beta.threads.runs.create(
			thread_id=thread.id,
			assistant_id=assistant.id
		)

		while run.status in ['queued', 'in_progress', 'cancelling']:
			time.sleep(1) # Wait for 1 second
			run = client.beta.threads.runs.retrieve(
			thread_id=thread.id,
			run_id=run.id
			)

		if run.status == 'completed': 
			messages = client.beta.threads.messages.list(thread_id=thread.id)
			cont = True
			for message in messages.data:
				if message.role == 'assistant' and cont:
					for content_block in message.content:
						if content_block.type == 'text' and cont:
							result = content_block.text.value
							print(result)
							df.loc[df.index==i, 'renovated'] = result
							processed[row['summary_id']] = result
							cont = False

		else:
			print('FAILED:',run.status)

		if i%10==0:
			print('Exporting:')
			df.to_stata(file, write_index=False, version=117)

if __name__=="__main__":
	main()
