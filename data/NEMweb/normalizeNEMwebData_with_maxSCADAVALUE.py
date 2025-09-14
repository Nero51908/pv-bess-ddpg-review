import os
import pandas as pd

def find_max_scadavalue(dir_name: str) -> float:
  max_value = None
  for root, _, files in os.walk(dir_name):
    for file in files:
      if file.endswith('.csv'):
        file_path = os.path.join(root, file)
        print(file_path)
        try:
          df = pd.read_csv(file_path)
          current_max = df['SCADAVALUE'].max()
          if max_value is None or current_max > max_value:
            max_value = current_max
          print(f"Found SCADAVALUE in {file_path}: {current_max}")
        except Exception as e:
          print(f"Error reading {file_path}: {e}")
  return max_value

def normalize_with_maxSCADAVALUE(dir_name: str, max_value: float) -> None:
  for root, _, files in os.walk(dir_name):
    for file in files:
      if file.endswith('.csv'):
        file_path = os.path.join(root, file)
        try:
          df = pd.read_csv(file_path)
          # backup the files
          backup_path = file_path.replace('.csv', '.csv.bak')
          df.to_csv(backup_path, index=False)
          # normalize SCADAVALUE and FORECAST_POE50 by max_value and write to csv
          df['SCADAVALUE'] /= max_value
          df['FORECAST_POE50'] /= max_value
          # df[['SCADAVALUE', 'FORECAST_POE50']] = df[['SCADAVALUE', 'FORECAST_POE50']] / max_value
          df.to_csv(file_path, index=False)
          print(f"Normalized {file_path}")
        except Exception as e:
          print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
  directory = 'RUGBYR1/202408_202505' #'PUT DUID HERE'
  max_scadavalue = find_max_scadavalue(directory)
  print(f"Max SCADAVALUE found: {max_scadavalue}")
  if max_scadavalue is not None:
    normalize_with_maxSCADAVALUE(directory, max_scadavalue)
  else:
    print("No SCADAVALUE found in the specified directory.")
  print("Normalization complete.")
