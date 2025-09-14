import os
import pandas as pd
import argparse
import numpy as np
def mape_loss_np(input:np.ndarray, target:np.ndarray):
  nominator   = np.abs(input - target)
  denominator = np.abs(input)
  try:
#   mape = np.sum(nominator / denominator) / target.size * 100
    mape = np.mean(nominator / denominator) * 100
  except:
    mape = float('nan')
  return mape
def display_mape(best_csv):
  df = pd.read_csv(best_csv)
  for index, row in df.iterrows():
    csv_path = os.path.join('evaluation', f"{row['dataset']}", f"{row['filename']}")
    if not os.path.exists(os.path.dirname(csv_path)):
      print(f'{csv_path} does not exist')
    df = pd.read_csv(csv_path)
    df = df[df['Pnet'] > 0]
    Pnet = df['Pnet'].to_numpy()
    Pdfc = df['Pdfc'].to_numpy()
    mape = mape_loss_np(Pnet, Pdfc)
    print(row['BESS_capacity'], csv_path, f'{mape=}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='calculate mape')
  parser.add_argument("--csv",    type=str, help="Best csv")
  args, _ = parser.parse_known_args()
  display_mape(args.csv)
