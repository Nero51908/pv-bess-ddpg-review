from collections import namedtuple
import os
import typing
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# note: do not import config in this module, as it may cause circular import

def get_current_time():
  return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

#==================== Unit Conversion =================#
def pus_to_puh(x):
  return x / 3_600

def puh_to_pus(x):
  return x * 3_600

def percent_to_fraction(x):
  return x / 100

def fraction_to_percent(x):
  return x * 100

#==================== Named Tuple for storing the results =================#
# namedtuple for upper (lim sup) and lower (lim inf) limits of the state space
StateSpaceLimits = namedtuple('StateSpaceLimits', ['limsup', 'liminf'])

#==================== Assigning Named Tuple =================#
def calc_Eb_boundary(Eb_max_pus, soc_boundary: StateSpaceLimits) -> StateSpaceLimits:
  lim_Eb = [percent_to_fraction(lim_in_soc) * Eb_max_pus for lim_in_soc in soc_boundary]
  # output is in pus
  return StateSpaceLimits(*lim_Eb)

def calc_Pb_boundary(Eb_max_puh, power_boundary: StateSpaceLimits) -> StateSpaceLimits:
  lim_Pb = [lim_in_Erate * Eb_max_puh for lim_in_Erate in power_boundary]
  # output is in pu
  return StateSpaceLimits(*lim_Pb)

#==================== check and create directories =================#
def create_directories_if_nonexistent(dir_names: typing.Dict[str, str]) -> None:
  # create paths used if they don't exist yet
  for the_path in dir_names.values():
    if not os.path.exists(the_path):
      os.makedirs(the_path)
      print(f"create_directories_if_nonexistent(): new directory ./{the_path} created.")

#=================== determine torch.device ====================#
def determine_default_tensor_operation_device():  
  # Use GPU or MPS if available. Otherwise use CPU (RAM).
  if torch.backends.mps.is_available():
    default_tensor_operation_device = torch.device('mps')
    print('Using Apple MPS as default_tensor_operation_device')

  elif torch.cuda.is_available():
      default_tensor_operation_device = torch.device('cuda')
      print('Using Nvidia CUDA as default_tensor_operation_device')
  else:
    default_tensor_operation_device = torch.device('cpu')
    print('No GPU, using CPU as default_tensor_operation_device')
  
  return default_tensor_operation_device

#============================= Performance Indicators ========================================#
def perfect_plan_rate(Pnet: np.ndarray, Pdfc: np.ndarray, atol: float):
  perfect_count = np.isclose(Pnet, Pdfc, atol=atol).sum()
  print(f"perfect_plan_rate(): {perfect_count=}, {Pnet.size=}")
  perfect_rate = perfect_count / Pnet.size * 100.
  return perfect_rate

def mean_curtailment(Pm: np.ndarray, actual_c: np.ndarray):
  curtailed_power = np.multiply(Pm, actual_c)
  mean_curtailed_power = np.mean(curtailed_power)
  return mean_curtailed_power # pu

def mean_curtailment_ratio(actual_c: np.ndarray):
  mean_actual_c = np.mean(actual_c)
  return mean_actual_c

def mse_loss_np(input: np.ndarray, target: np.ndarray):
  mse = np.mean((input - target)**2)
  return mse

def rmse_loss_np(input: np.ndarray, target: np.ndarray):
  mse = mse_loss_np(input, target)
  rmse = np.sqrt(mse)
  return rmse

def mae_loss_np(input:np.ndarray, target:np.ndarray):
  abs_diff = np.abs(input - target)
  mae = np.mean(abs_diff)
  return mae

def mape_loss_np(input:np.ndarray, target:np.ndarray):
  nominator   = np.abs(input - target)
  denominator = np.abs(input)
  try:
#   mape = np.sum(nominator / denominator) / target.size * 100
    mape = np.mean(nominator / denominator) * 100
  except:
    mape = float('nan')
  return mape

#==================== Data Preprocessing ==================#
def beginning_of_nonzero_cluster_indices(nonzero_indices):

  # Find the difference between consecutive nonzero indices
  diff = np.diff(nonzero_indices)

  # Identify the start of a new cluster (where the difference is greater than 1)
  cluster_starts = np.where(diff > 1)[0] + 1

  beginning_indices_of_clusters = np.insert(nonzero_indices[cluster_starts], 0, nonzero_indices[0])
  
  return beginning_indices_of_clusters

def flatten_nested_dict_to_dict(nested_dict) -> dict:
  flattened_data = {}
  for key, value in nested_dict.items():
    if isinstance(value, dict):
      for subkey, subvalue in value.items():
        flattened_data[f"{key}_{subkey}"] = subvalue
    else:
        flattened_data[key] = value
  
  return flattened_data

#==================== Visualize Evaluation ==================#
def visualize_pnet_pdfc(csv_file_path: str, target_dir: str, title: str, save: bool = True, show: bool = False, show_datetime: bool = False):
    print(f"visualize_pnet_pdfc(): {csv_file_path}, {save=}, {show=}.")
    df = pd.read_csv(csv_file_path)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,)# subplot_titles=["Power", "Cost",])
    fig.update_layout(
      # title={'text': title, 'x': 0, 'xanchor': 'left'},
      legend=dict(font=dict(size=14)),
      margin=dict(l=4, r=4, t=4, b=4),
      title={'text': '', 'x': 0, 'xanchor': 'left'},
      template='ggplot2',
    )
    if show_datetime:
      x_data = df["env_state_datetime"]
      bar_offset = 0
      bar_width = None
    else:
      x_data = df.index
      bar_offset = 0
      bar_width = None
    # 1st row 
    fig.update_yaxes(title_text="Power (pu)", range=[0,1], row=1, col=1)
    fig.add_trace(go.Bar(x=x_data, y=df["reward"], name="Reward", marker_color="magenta", offset=bar_offset, width=bar_width, opacity=0.2), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=df["env_state_pv_potential"], name=r"$P_\text{m}$", line_color="orange", line_shape="hv"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=df["env_state_pv_forecast"], name=r"$P_\text{f}$", line_color="cyan", line_shape="hv"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=df["Pnet"], name=r"$P_\text{net}$", line_color="red", line_shape="hv"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=df["Pdfc"], name=r"$P_\text{DFC}$", line_color="blue", line_shape="hv", line_width=1), row=1, col=1)

    # 2nd row 
    fig.update_yaxes(title_text="SoC", range=[0,1], row=2, col=1)
    fig.update_xaxes(title_text="Simulation step", row=2, col=1) #range=[4949,5370]
    fig.add_trace(go.Scatter(x=x_data, y=df["env_state_initial_soc"]/100, fill="tozeroy", name="SoC", line_color="limegreen", line_shape="spline"), row=2, col=1)

    if save:
      if not os.path.exists(target_dir):
        os.makedirs(target_dir)
      save_path = os.path.join(target_dir, os.path.basename(csv_file_path).split('.csv')[0])
      save_path = f"{save_path}.html"
#      fig.write_html(save_path)
      fig.write_html(f"{save_path}")
      # fig.write_image(f"{save_path}.pdf", width=1000, height=300)
      print(f"visualize_pnet_pdfc(): {save_path} saved.")
      
    if show:  
      fig.show()

def visualize_pdfc_vs_c(csv_file_path: str):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Create a Plotly Express scatter plot
    fig = px.scatter(
        df,
        x="curtailment",
        y="Pdfc",
        color_discrete_sequence=["red"],
        labels={
            "curtailment": "Curtailment",
            "Pdfc": "Pdfc",
        },
        title="Pdfc vs Curtailment (Scatter)"
    )

    # add a linear regression trendline:
    fig = px.scatter(df, x="curtailment", y="Pdfc", trendline="ols")
    fig.show()

def visualize_pdfc_vs_pf(csv_file_path: str):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Create a Plotly Express scatter plot
    fig = px.scatter(
        df,
        x="env_state_pv_forecast",
        y="Pdfc",
        color_discrete_sequence=["red"],
        labels={
            "env_state_pv_forecast": "Pf",
            "Pdfc": "Pdfc",
        },
        title="Pdfc vs Pf"
    )

    # add a linear regression trendline:
    fig = px.scatter(df, x="env_state_pv_forecast", y="Pdfc", trendline="ols")
    fig.show()    
