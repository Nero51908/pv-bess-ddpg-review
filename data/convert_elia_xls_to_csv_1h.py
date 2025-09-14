'''
This script will take the mean of every 4 rows in the data columns.
'''
import os
import pandas as pd
dataset = 'Elia_2016-2018'
for dirpath, _, filenames in os.walk(dataset):
  for filename in filenames:
    if filename.endswith('.xls'):
      file_path = os.path.join(dirpath, filename)
      df = pd.read_excel(file_path, header=3)
      base = df.iloc[4,6]
      extracted_data = df.groupby(df.index//4)[['Most recent forecast [MW]', 'Day-Ahead forecast [MW]', 'Corrected Upscaled Measurement [MW]']].mean() / base
      extracted_data.insert(0, 'DateTime', df['DateTime'][::4].reset_index(drop=True))
      extracted_data.to_csv(file_path.replace('.xls', '.csv'), index=False)
      print(f"Converted {file_path} to CSV format.")
