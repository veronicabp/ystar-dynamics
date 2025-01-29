#%%
import os 
import numpy as np
import pandas as pd

data_folder="../data/original"
area = 'NW'
files = [file for file in os.listdir(os.path.join(data_folder, 'working')) if file.startswith(f'bootstrap_ystar_{area}')]

estimates = np.array([])
for file in files:
    df = pd.read_csv(os.path.join(data_folder, 'working', file))
    estimates = np.concatenate((estimates, np.array(df['ystar_estimates'])))

print('Bootstrap SE:', estimates.std())
print('Num bootstraps:', len(estimates))

for n in range(5,len(estimates)):
    print('Bootstrap SE:', estimates[0:n].std())
    print('Num bootstraps:', n)