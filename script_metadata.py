import os
import pandas as pd
import config
import definitions

def compute_metadata(pd_series):
  min_value = pd_series.min().round(3)
  max_value = pd_series.max().round(3)
  mean_value= pd_series.mean().round(3)
  std_value = pd_series.std().round(3)
  metadata_dict = {
    'min': min_value,
    'max': max_value,
    'mean': mean_value,
    'std': std_value
  }
  return metadata_dict

def journalize_metadata(dataset:str):
  columns_of_interest = ['Pf', 'Pm']
  data_path    = os.path.join(config.data_register[dataset]['path'])
  data_feature = config.data_register[dataset]['feature']
  
  data_buffer = definitions.Data_Buffer()
  data_buffer.read_data_from(data_path, *data_feature['file_structure'])
  data = data_buffer.prepare_data_pd()
  for column in columns_of_interest:
    metadata_dict = {
      'dataset': dataset,
      'column': column,
    }
    metadata_dict.update(compute_metadata(data[column]))
    metadata_df = pd.DataFrame([metadata_dict])
    metadata_df.to_csv('metadata.csv', mode='a', header=not os.path.exists('metadata.csv'), index=False)

if __name__ == "__main__":
  datasets_of_interest = ['Elia_2016-2018_1h_dah', 'Elia_2016-2018_1h_hah', 'Elia_2019_1h_dah', 'Elia_2019_1h_hah'] #'RUGBYR1_old', 'RUGBYR1_new', 'BANN1_old', 'BANN1_new', 'EDENVSF1_old', 'EDENVSF1_new']
  for dataset in datasets_of_interest:
    journalize_metadata(dataset)
