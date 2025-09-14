# Python standard lib
from collections import namedtuple, OrderedDict
import copy
import random
import logging
import os
# from multiprocessing import Pool

# additional modules
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import wandb

# config
import config

#=================== namedtuple defined for readability =======================#
# One experience is (o, a, r, o'), o and o' are physical observation
# each field may hold an iterator like a tensor.

# if commenting out this one does not trigger a bug, delete it
# Package_for_Q_Net_Input\
# = namedtuple('Package_for_Q_Net_Input', ['action',\
#                                          'observation_with_embedding'])

Experience\
= namedtuple('Experience', ['observation',\
                            'action',\
                            'reward',\
                            'd',\
                            'next_observation'])
# ,\
                            # 'next_action'])

Physical_Observation\
= namedtuple('Physical_Observation', ['forecast',\
                                      'state_of_charge'])

# In this context, o is (o_forecast, o_battery_energy)
# o' is (o'_forecast, o'_battery_energy).
# Simulation usually requires (o, a) to obtain (r, o')
# s is the ground truth PV power for the time that o_forecast covers.
# s is needed for the simulation.
# (o_forecast, o_battery_energy), a, o'_forecast and s are known,
# use simulation to find (r, o'_battery_energy).
Environment_Setup\
= namedtuple('Environment_Setup', ['observation_forecast',\
                                   'next_observation_forecast',\
                                   'measured_PV_power'])

# named tuple to represent battery power limit and energy limit. ref: https://docs.python.org/3/library/collections.html#collections.namedtuple
Upper_and_lower_limit\
= namedtuple('Upper_and_lower_limit', ['upper_limit',\
                                        'lower_limit'])

#=================== data buffer definition =================================#
Forecast_and_Measured\
= namedtuple('Forecast_and_Measured', ['forecast',\
                                       'measured'])


#==============================================================================#
Loss_Daq_Data\
= namedtuple('Loss_Daq_Data', ['generation',\
                               'loss'])

#==================== zero mean fixed variation normalization =================#
def zero_mean_fixed_variation_normalize(tensor: torch.Tensor, std_dev):
  # get mean and std_dev of the tensor
  mean = tensor.mean()
  original_std = tensor.std()
  # normalize the tensor
  return (tensor - mean) / original_std * std_dev


#==================== data input buffer ======================================#
class Data_Buffer:
  def __init__(self,
               device=torch.device('cpu')
              ):
    
    self.device        = device
    self.list_to_stack = []
    self.list_of_darrays = []
    self.list_of_tarrays = []
    self.power_base        = None
    self.perfect_threshold = None

  def __len__(self):
    return len(self.list_of_days)

  def _load_files_to_df(self, filepath, header_row_number: int, column_names_to_read: list[str], data_format: str):
    if data_format == 'NEM_csv':
      df = pd.read_csv(filepath, header=header_row_number, parse_dates=["DATETIME"])\
              .sort_values(by=["DATETIME"])
      power_base = 1 # WARNING: NEMweb data should be normalized already. 
    elif data_format == 'Elia_xls':
      df = pd.read_excel(filepath, header=header_row_number, parse_dates=["DateTime"], date_format='%d/%m/%Y %H:%M')\
              .sort_values(by=["DateTime"])
      power_base = df.iloc[4, 6] # WARNING: hard-coded position of power base for Elia xls data
    elif data_format == 'Elia_csv':
      df = pd.read_csv(filepath, header=header_row_number, parse_dates=["DateTime"], date_format='%d/%m/%Y %H:%M')\
              .sort_values(by=["DateTime"])
      power_base = 1 # WARNING: Elia csv data should be normalized already.
    elif data_format == 'DeepComp_csv':
      df = pd.read_csv(filepath, header=header_row_number, parse_dates=["Time"], date_format='%d/%m/%Y %H:%M')\
              .sort_values(by=["Time"])
      power_base = df[column_names_to_read[-1]].max()
    else:
      raise ValueError(f"DataBuffer.read_data_from(): Failed to read {filepath} using {data_format} format.")
    self.power_base = power_base
    return df, power_base

  """
  How to use read_data_from() and the data_register dictionary:
  >>> db2 = Data_Buffer()
  >>> data_path2 = config.data_register[dataset2]['path']
  >>> data_feature2 = config.data_register[dataset2]['feature']
  >>> db2.read_data_from(data_path2, *data_feature2['file_structure'])
  >>> darray2 = np.concatenate(db2.list_of_darrays)
  """
  def read_data_from(self, input_path: str, 
                     header_row_number: int, 
                     points_per_day: int, 
                     column_names_to_read: list[str], 
                     data_format: str):
    # Assert that the input path exists
    assert os.path.exists(input_path), f"Input path {input_path} does not exist."

    # Read data from the input path
    print(f"DataBuffer.read_data_from(): Reading data from {input_path}...")
    for dirpath, _, filenames in os.walk(input_path):
      for filename in filenames:
        # skip hidden files
        if filename.startswith('.'):
          continue 
        # skip files that are not csv or xls
        if not (filename.endswith('.csv') or filename.endswith('.xls')):
          print(f"DataBuffer.read_data_from(): Skipped {filename} (not csv or xls).")
          continue
        # read data as a dataframe from csv or xls (currently supports data from NEM(AEMO) or Elia)
        filepath = os.path.join(dirpath, filename)
        df, power_base = self._load_files_to_df(filepath, header_row_number, column_names_to_read, data_format)
        pfpm      = df[column_names_to_read[1:]].to_numpy()
        timestamp = df[column_names_to_read[0]].to_numpy()

        try:
          pfpm_days      = pfpm.reshape(-1, points_per_day, pfpm.shape[1])
          timestamp_days = timestamp.reshape(-1, points_per_day)
        except:
          print(f'DataBuffer.read_data_from(): Skipped {filename} (failed to reshape it as {(-1, points_per_day, len(column_names_to_read))})')
          continue
        
        # remove the days that contain NaN; 
        wanted_days_bool = ~np.isnan(pfpm_days).any(axis=(1, 2))
        pfpm_days        = pfpm_days[wanted_days_bool] / power_base # normalize the data
        timestamp_days   = timestamp_days[wanted_days_bool]
        
        pfpm_array       = pfpm_days.reshape(-1, pfpm_days.shape[2])
        timestamp_array  = timestamp_days.reshape(-1)
        
        self.list_of_darrays.append(pfpm_array) 
        self.list_of_tarrays.append(timestamp_array)

  def prepare_data_pd(self) -> pd.DataFrame:
    darray = np.concatenate(self.list_of_darrays)
    tarray = np.concatenate(self.list_of_tarrays)
    
    df = pd.DataFrame(darray, columns=['Pf', 'Pm'])
    df.insert(0, 't', tarray)
    df = df.sort_values(by=['t'])
    return df
  
  def prepare_data_np(self) -> np.ndarray:
    # darray = np.concatenate(self.list_of_darrays)
    # tarray = np.concatenate(self.list_of_tarrays)
    
    # df = pd.DataFrame(darray, columns=['Pf', 'Pm'])
    # df.insert(0, 't', tarray)
    # df = df.sort_values(by=['t'])
    df = self.prepare_data_pd()
    return df.to_numpy()
  
  '''
  Parameters are loaded by expanding a tuple, like this:
  data_path    = os.path.join(config.data_register[dataset]['path'])
  data_feature = config.data_register[dataset]['feature']
  read_data_from(data_path, *data_feature["file_structure"])
  '''
  # hard-coded for Elia data
  def read_data(self,
                input_path: str, 
                header_row_number: int, 
                points_per_day: int, 
                column_names_to_read: list[str], 
                data_format: str):
    for dirpath, _, filenames in os.walk(input_path):
      for filename in filenames:
        # skip hidden files
        if filename.startswith('.'):
          continue 
        # skip files that are not csv or xls
        if not (filename.endswith('.csv') or filename.endswith('.xls')):
          print(f"DataBuffer.read_data_from(): Skipped {filename} (not csv or xls).")
          continue
        # elia data header starts from row3 (row No. in df starts from 0. Row 0,1,2 are front matter)
        logging.info(f'Data_Buffer.read_data(): Reading {filename}')
        filepath = os.path.join(dirpath, filename)
        df, power_base = self._load_files_to_df(filepath, header_row_number, column_names_to_read, data_format)
        logging.info(f'Data_Buffer.read_data(): {power_base=} (Mw)')

        # extract data from dataframe and then assign tensor x
        # print('read_data():', column_names_to_read[1::]) # debugging
        tensor_x = torch.tensor(df[column_names_to_read[1::]].values, dtype=torch.float32)
        logging.info('Data_Buffer.read_data(): Raw data forms tensor_x of size %s', tensor_x.shape)
        
        # view the input tensor as (day, data points per data, pfpm)
        try:
          tensor_x_all_days = tensor_x.view(-1, points_per_day, 2)
          logging.info(f'Data_Buffer.read_data(): Successfully read {tensor_x_all_days.shape=} (day, data points per day, pfpm).')
        except:
          logging.warning(f'Data_Buffer.read_data(): Unable to view {filename} as size(-1,{points_per_day},2).')
          continue
        
        # filter out days that contain NaN; 
        tensor_x_all_days = tensor_x_all_days[(~torch.isnan(tensor_x_all_days[::,::,0])).all(dim=1)|\
                                              (~torch.isnan(tensor_x_all_days[::,::,1])).all(dim=1),::,::]\
                                               / power_base
        logging.info('Data_Buffer.read_data(): After NaN filter, data tensor shape is %s', tensor_x_all_days.shape)
        
        self.list_to_stack.extend(tensor_x_all_days) # PyTorch tensors are iterables through their dim=0. extend() accepts iterables.
  
  @torch.no_grad()
  def unfold_according_to_size(self, size: int, test_flag=False) -> torch.Tensor:
    if test_flag:
      data_for_testing = torch.stack(self.list_to_stack)
      return data_for_testing.unfold(1, size, config.next_forecast_produced_after_N).to(self.device)
    else:
      split_point   = config.training_data_portion * len(self.list_to_stack) // 100
      random.shuffle(self.list_to_stack)
      data_tensor = torch.stack(self.list_to_stack)
      data_for_training, data_for_validating = torch.split(data_tensor, [split_point, len(data_tensor) - split_point])
      # a quick tip for Tensor.unfold(): Tensor.unfold(dimension, size, step)
      return data_for_training.unfold(1, size, config.next_forecast_produced_after_N).to(self.device),\
            data_for_validating.unfold(1, size, config.next_forecast_produced_after_N).to(self.device)
    
#=================== replay buffer definition =================================#
class Replay_Buffer:
  def __init__(self, training_tensor_shape, capacity, batch_device: torch.device, storage_device=torch.device('cpu')):
    self.soft_capacity_limit = capacity
    self.batch_device = batch_device
    self.storage_device = storage_device
    # initialize
    training_tensor_shape= list(training_tensor_shape)
    training_tensor_shape.remove(2) # remove value 2 from the list. 2 is the length of the dimension that distinguishes forecast and measured data.
    observation_shape    = training_tensor_shape.copy()
    action_shape         = training_tensor_shape.copy()
    reward_shape         = training_tensor_shape.copy()
    # modify the shapes information for creating place holders.
    observation_shape[-1]= config.forecast_length_per_observation + config.soc_length
    reward_shape[-1]     = 1
    observation_shape[1]-= 1
    action_shape[1]     -= 1
    reward_shape[1]     -= 1

    # significant experiences where batch for RL agent training will be sampled from
    self.significant_observation_T         = torch.tensor([], dtype=torch.float32, device=self.storage_device)
    self.significant_action_T              = torch.tensor([], dtype=torch.float32, device=self.storage_device)
    self.significant_reward_T              = torch.tensor([], dtype=torch.float32, device=self.storage_device)
    self.significant_d_tensor              = torch.tensor([], dtype=torch.float32, device=self.storage_device)
    self.significant_observation_T_plus_one= torch.tensor([], dtype=torch.float32, device=self.storage_device)
  
  def memorize(self,\
               observation_T_tensor: torch.Tensor,\
               action_tensor: torch.Tensor,\
               reward_tensor: torch.Tensor,\
               observation_T_plus_one_tensor: torch.Tensor):
    # input tensors should be in GPU
    
    # Process the experience to filter out significant ones.
    # Create a mask where all elements (except for the last one) in the last dimension are not zero.
    #
    # First, locate the significant experiences.
    # the significants are experiences derived from non-zero forecast time series observed at time T.
    T_none_zero_row_mask          = torch.all(observation_T_tensor[...,:-1:] >= config.zero_power_threshold, dim=-1, keepdim=True)
    T_plus_one_none_zero_row_mask = torch.all(observation_T_plus_one_tensor[...,:-1:] >= config.zero_power_threshold, dim=-1, keepdim=True)
    
    # Get the shape of the tensors.
    observation_shape = observation_T_tensor.shape
    action_shape      = action_tensor.shape
    reward_shape      = reward_tensor.shape
    
    # Then, expand the True values to the size of that dimension in the tensors.
    observation_shape_none_zero_row_mask = T_none_zero_row_mask.expand(-1,-1,-1, observation_shape[-1])
    action_shape_none_zero_row_mask      = T_none_zero_row_mask.expand(-1,-1,-1, action_shape[-1])
    reward_shape_none_zero_row_mask      = T_none_zero_row_mask.expand(-1,-1,-1, reward_shape[-1])
    observation_T_plus_one_shape_none_zero_row_mask = T_plus_one_none_zero_row_mask.expand(-1,-1,-1, observation_shape[-1])

    # Finally, use the boolean mask to index the tensors for the significant experiences.
    # the significants are viewed as 2D tensors: dim0 counts the data entry, dim1 indexes the data in time series position.
    significant_observation_T = observation_T_tensor[observation_shape_none_zero_row_mask].view(-1,observation_shape[-1])
    significant_action_T      = action_tensor[action_shape_none_zero_row_mask].view(-1,action_shape[-1])
    significant_reward_T      = reward_tensor[reward_shape_none_zero_row_mask].view(-1,reward_shape[-1])
    significant_reward_T_normalized = zero_mean_fixed_variation_normalize(significant_reward_T, std_dev=1.0)
    significant_observation_T_plus_one = observation_T_plus_one_tensor[observation_T_plus_one_shape_none_zero_row_mask].view(-1,observation_shape[-1])

    # also need to locate the terminal states in the observation T plus One tensor.
    # for each day, there is only one terminal state. 
    # that is the one after the last observation T with non-zero forecast time series.
    # find the indices of the last states using the forecasting part of the observation tensor.
    # Flip the tensor along the dimension that represents the time in the day.
    flipped_T_plus_one_none_zero_row_mask = torch.flip(T_plus_one_none_zero_row_mask, dims=[1])
    # Find the position of the last True value in each row of the dimension along which the tensor was flipped.
    true_indices = torch.argmax(flipped_T_plus_one_none_zero_row_mask.int(), dim=1)

    # Create a tensor of zeros with the same shape as T_plus_one_none_zero_row_mask
    flipped_d_tensor = torch.zeros_like(T_plus_one_none_zero_row_mask, dtype=torch.float32, device=self.batch_device).bool()

    # Get the indices of the first, second, fourth, and fifth dimensions
    indices_dim0, indices_dim2, indices_dim3 = torch.meshgrid(
        torch.arange(flipped_T_plus_one_none_zero_row_mask.size(0)), 
        torch.arange(flipped_T_plus_one_none_zero_row_mask.size(2)),
        torch.arange(flipped_T_plus_one_none_zero_row_mask.size(3)),indexing='ij'
    )

    # Use these indices to index into flipped_T_plus_one_none_zero_row_mask
    flipped_d_tensor[indices_dim0, true_indices, indices_dim2, indices_dim3] = True
    d_tensor      = torch.flip(flipped_d_tensor, dims=[1])
    significant_d_tensor = d_tensor[T_plus_one_none_zero_row_mask].view(-1,reward_shape[-1])

    # move tensors to storage device and calculate normalized reward
    significant_observation_T = significant_observation_T.to(self.storage_device)
    significant_action_T      = significant_action_T.to(self.storage_device)
    significant_reward_T      = significant_reward_T.to(self.storage_device)
    significant_reward_T_normalized = significant_reward_T_normalized.to(self.storage_device)
    significant_d_tensor      = significant_d_tensor.to(self.storage_device)
    significant_observation_T_plus_one = significant_observation_T_plus_one.to(self.storage_device)

    # Assign the significant experiences to the self.tensors
    if self.len() <= config.replay_buffer_capacity:
      self.significant_observation_T = torch.cat((self.significant_observation_T, significant_observation_T))
      self.significant_action_T = torch.cat((self.significant_action_T, significant_action_T))
      self.significant_d_tensor = torch.cat((self.significant_d_tensor, significant_d_tensor))
      self.significant_observation_T_plus_one = torch.cat((self.significant_observation_T_plus_one, significant_observation_T_plus_one))
      if config.epoch_reward_normalization:
        self.significant_reward_T = torch.cat((self.significant_reward_T, significant_reward_T_normalized))
      else:
        self.significant_reward_T = torch.cat((self.significant_reward_T, significant_reward_T))
    else: 
      self.significant_observation_T = torch.cat((self.significant_observation_T[significant_observation_T.shape[0]::], significant_observation_T))
      self.significant_action_T = torch.cat((self.significant_action_T[significant_action_T.shape[0]::], significant_action_T))
      self.significant_d_tensor = torch.cat((self.significant_d_tensor[significant_d_tensor.shape[0]::], significant_d_tensor))
      self.significant_observation_T_plus_one = torch.cat((self.significant_observation_T_plus_one[significant_observation_T_plus_one.shape[0]::], significant_observation_T_plus_one))
      if config.epoch_reward_normalization:
        self.significant_reward_T = torch.cat((self.significant_reward_T[significant_reward_T_normalized.shape[0]::], significant_reward_T_normalized))
      else:
        self.significant_reward_T = torch.cat((self.significant_reward_T[significant_reward_T.shape[0]::], significant_reward_T))

  def len(self):
    return self.significant_observation_T.shape[0]
  
  def randomly_sample_a_batch(self, batch_size):
    # select a batch from the significants randomly
    batch_idx = torch.randperm(self.significant_observation_T.shape[0], device=self.storage_device)[:batch_size]

    # return the batch which is hopefullty stored in a GPU
    significant_experience =  Experience(self.significant_observation_T[batch_idx,::].to(self.batch_device),\
                                         self.significant_action_T[batch_idx,::].to(self.batch_device),\
                                         self.significant_reward_T[batch_idx,::].to(self.batch_device),\
                                         self.significant_d_tensor[batch_idx,::].to(self.batch_device),\
                                         self.significant_observation_T_plus_one[batch_idx,::].to(self.batch_device))
    return significant_experience
    
#===================== Environment Dynamics ==================#
class Battery:
  def __init__(self,\
               battery_capacity_pus,\
               soc_upper_limit,\
               soc_lower_limit,\
               soc_e_to_fcharge,\
               soc_f_to_echarge,\
               charging_efficiency,\
               discharging_efficiency,\
               time_per_step_sec,
               device=torch.device('cpu')
              ):
    # specify the device to store the tensors defined in this class
    self.device = device

    # battery limits in terms of energy (per unit * sec)
    # and because I heard CPU deals with serial tasks better than GPU, create a clone of the tensor on cpu for time seriese smimulation in validation code.
    self.battery_capacity_pus  = torch.tensor(battery_capacity_pus, dtype=torch.float32, device=self.device) # value in per unit sec
    self.battery_capacity_pus_cpu = self.battery_capacity_pus.clone().to('cpu')

    self.battery_soc_limit = Upper_and_lower_limit(upper_limit=torch.tensor(soc_upper_limit, dtype=torch.float32, device=self.device),\
                                                   lower_limit=torch.tensor(soc_lower_limit, dtype=torch.float32, device=self.device))
    self.battery_soc_limit_cpu = Upper_and_lower_limit(upper_limit=torch.tensor(soc_upper_limit, dtype=torch.float32, device=torch.device('cpu')),\
                                                       lower_limit=torch.tensor(soc_lower_limit, dtype=torch.float32, device=torch.device('cpu')))
    
    self.battery_energy_operational_boundary_pus = Upper_and_lower_limit(upper_limit=self.battery_capacity_pus * self.battery_soc_limit.upper_limit / 100,\
                                                                         lower_limit=self.battery_capacity_pus * self.battery_soc_limit.lower_limit / 100)
    self.battery_energy_operational_boundary_pus_cpu = Upper_and_lower_limit(upper_limit=self.battery_capacity_pus_cpu * self.battery_soc_limit_cpu.upper_limit / 100,\
                                                                             lower_limit=self.battery_capacity_pus_cpu * self.battery_soc_limit_cpu.lower_limit / 100)

    # battery limits in terms of power (per unit)
    # positve battery power means charging; negative power means it's discharging. 
    # unit for battery power is p.u. meaning per unit PV installed capacity in terms of power.
    self.battery_power_limit_pu = Upper_and_lower_limit(upper_limit = battery_capacity_pus / soc_e_to_fcharge,\
                                                        lower_limit = -1 * battery_capacity_pus / soc_f_to_echarge)
    self.battery_power_limit_pu_cpu = Upper_and_lower_limit(upper_limit = self.battery_capacity_pus_cpu / soc_e_to_fcharge,\
                                                            lower_limit = -1 * self.battery_capacity_pus_cpu / soc_f_to_echarge)

    # battery properties
    self.charging_efficiency        = torch.tensor(charging_efficiency, dtype=torch.float32, device=self.device) # decimal number [0,1]
    self.charging_efficiency_cpu    = self.charging_efficiency.clone().to('cpu') # decimal number [0,1]
    self.discharging_efficiency     = torch.tensor(discharging_efficiency, dtype=torch.float32, device=self.device) # decimal number [0,1]
    self.discharging_efficiency_cpu = self.discharging_efficiency.clone().to('cpu') # decimal number [0,1]
    # simulator properties
    self.time_per_step_sec = torch.tensor(time_per_step_sec, dtype=torch.float32, device=self.device) # sec
    self.time_per_step_sec_cpu = self.time_per_step_sec.clone().to('cpu') # sec



    # print initial values
    print("Battery model created with the following properties:")
    print(f"{self.battery_capacity_pus=}")
    print(f"{self.battery_soc_limit=}")
    print(f"{self.time_per_step_sec=}")

  # control_plan should be the same length as available_PV_power.
  # Both are in terms of power (per unit PV installed capacity).
  # They also have the same time resolution.
  # !! Important Assumption: This simulation assumes constant power during a time step
  # Key Reference Paper: This simulation uses the battery model in DeepComp.
  # This battery uses Pd and Pc to represent battery power, whereas I use one universal variable, battery power: positve battery power means charging; negative power means it's discharging. 
  # Auto grad setting: mathematical operation in simulation not to be tracked by auto grad 
  # If the input tensors have 4 dimensions,
  # dim0 for the number of days; dim1 for the time in a day; dim2 for different SoC levels; dim3 for data in one forecast horizon.
  # The fundamental assumtion that sumulation() makes is that the input tensors have the last dimension the data for computation.
  @torch.no_grad()
  def simulate(self,\
               control_plan_tensor: torch.Tensor,\
               available_PV_power_tensor: torch.Tensor,\
               initial_soc_tensor: torch.Tensor):

    # create placehodlers for output data from simulation
    power_tensor_shape      = control_plan_tensor.shape
    energy_tensor_shape_as_list = list(power_tensor_shape)
    energy_tensor_shape_as_list[-1] += 1 # energy has one more time step than power
    energy_tensor_shape     = torch.Size(energy_tensor_shape_as_list)      
    battery_power_tensor    = torch.empty_like(control_plan_tensor, dtype=torch.float32, device=self.device)
    net_output_power_tensor = torch.empty_like(control_plan_tensor, dtype=torch.float32, device=self.device)
    battery_energy_tensor   = torch.empty(energy_tensor_shape, dtype=torch.float32, device=self.device)
    battery_soc_tensor      = torch.empty_like(battery_energy_tensor, dtype=torch.float32, device=self.device)

    # first, assign the initial state of charge to the first time step of the battery energy tensor.
    # then, the computation for simulation starts:
    battery_energy_tensor[...,0] = initial_soc_tensor.squeeze(-1) * self.battery_capacity_pus / 100
    
    for i in range(control_plan_tensor.shape[-1]):
      # deepcomp equation #(2a) P_climit = min(P_cmax, (1/eff_c)*(E_max*soc_max - E)/tdelta)
      battery_charging_power_limit_pu_tensor = torch.ones_like(battery_energy_tensor[...,i], dtype=torch.float32, device=self.device) * self.battery_power_limit_pu.upper_limit
      # charging power needed if the battery will be full in one forecast resolution time.
      # per unit * sec (energy) / sec = per unit (power)
      power_pu_to_fully_charge_battery_tensor = (1/self.charging_efficiency)\
                                                * (self.battery_energy_operational_boundary_pus.upper_limit - battery_energy_tensor[...,i])\
                                                / self.time_per_step_sec

      # maximum charging power allowed by the battery
      charging_power_pu_limit_tensor = torch.minimum(battery_charging_power_limit_pu_tensor, power_pu_to_fully_charge_battery_tensor)

      # equation #(2b) P_dlimit = min(P_dmax, eff_d*(E - E_max*soc_min)/tdelta)
      battery_discharg_power_limit_pu_tensor = torch.ones_like(battery_energy_tensor[...,i], dtype=torch.float32, device=self.device) * abs(self.battery_power_limit_pu.lower_limit)
      power_pu_to_deplete_battery_tensor = self.discharging_efficiency\
                                           * (battery_energy_tensor[...,i] - self.battery_energy_operational_boundary_pus.lower_limit)\
                                           / self.time_per_step_sec
      discharg_power_pu_limit_tensor = torch.minimum(battery_discharg_power_limit_pu_tensor, power_pu_to_deplete_battery_tensor)


      # a scalar tensor for value 0 in tensor operataions
      zero_tensor = torch.zeros_like(battery_energy_tensor[...,i], dtype=torch.float32, device=self.device)
      # equation (3a) P_c = min(max(P_pv_available - P_net_pv_battery_prediction, 0), P_climit)
      battery_charging_power_pu_tensor = torch.minimum(\
                                                      torch.maximum(available_PV_power_tensor[...,i] - control_plan_tensor[...,i], zero_tensor),\
                                                      charging_power_pu_limit_tensor)

      # equation (3b) P_d = min(max(P_net_pv_battery_prediction - P_pv_available, 0), P_dlimit)
      battery_discharg_power_pu_tensor = torch.minimum(\
                                                      torch.maximum(control_plan_tensor[...,i] - available_PV_power_tensor[...,i], zero_tensor),\
                                                      discharg_power_pu_limit_tensor)

      battery_power_tensor[...,i] = battery_charging_power_pu_tensor - battery_discharg_power_pu_tensor

      # equation (4) E_prime = E + eff_c * P_c * tdelta - (1 / eff_d) * P_d * tdelta
      battery_energy_tensor[...,i+1] = battery_energy_tensor[...,i]\
                                            + self.charging_efficiency * battery_charging_power_pu_tensor * self.time_per_step_sec\
                                            - (1/self.discharging_efficiency) * battery_discharg_power_pu_tensor * self.time_per_step_sec
      
      # P_net_pv_battery = P_pv_available - P_c + P_d
      net_output_power_tensor[...,i] = available_PV_power_tensor[...,i]\
                                            - battery_charging_power_pu_tensor\
                                            + battery_discharg_power_pu_tensor

    battery_soc_tensor = battery_energy_tensor / self.battery_capacity_pus * 100

    return net_output_power_tensor, battery_power_tensor, battery_soc_tensor

#============== Definition of Roles (neural networks) in RL =================#
class Actor(nn.Module):
  def __init__(self, forecast_length, additional_info_length, hidden_size, action_length, device=torch.device('cpu')):
    super().__init__()
    self.device = device
    # input to the actor network is called "observation"
    self.observation_size = forecast_length + additional_info_length
    # output (inference) from the actor nwetwork is called "action"
    self.inference_length = action_length

    self.neural_network = nn.Sequential(OrderedDict([
                                       ('Actor_linear_1', nn.Linear(self.observation_size, hidden_size, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Actor_Elu_1', nn.ELU()),
                                       ('Actor_linear_2', nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Actor_Elu_2', nn.ELU()),
                                       ('Actor_linear_3', nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Actor_Elu_3', nn.ELU()),
                                       ('Actor_linear_4', nn.Linear(hidden_size, action_length, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Actor_Elu_4', nn.ELU())
                                     ]))

  def forward(self, observation_tensor: torch.Tensor) -> torch.Tensor:
    actor_output = self.neural_network(observation_tensor)
    return actor_output

class Critic(nn.Module):
  def __init__(self, actor_observation_size, actor_action_length, hidden_size, device=torch.device('cpu')):
    super().__init__()
    self.device = device
    self.input_size = actor_observation_size + actor_action_length
    self.inference_size = 1
    self.neural_network = nn.Sequential(OrderedDict([
                                       ('Critic_linear_1', nn.Linear(self.input_size, hidden_size, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Critic_LeakyReLU_1', nn.LeakyReLU()),
                                       ('Critic_linear_2', nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Critic_LeakyReLU_2', nn.LeakyReLU()),
                                       ('Critic_linear_3', nn.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Critic_LeakyReLU_3', nn.LeakyReLU()),
                                       ('Critic_linear_4', nn.Linear(hidden_size, self.inference_size, bias=True, dtype=torch.float32, device=self.device)),
                                       ('Critic_LeakyReLU_4', nn.LeakyReLU())
                                     ]))

  def forward(self, what_actor_model_sees: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # dim0 is the index of experience in this batch
    # dim1 contains data in each experience 
    # Here, specify dim=1 for cat() so concatenation happens to dim=1.                                          
    critic_observation = torch.cat((what_actor_model_sees, action), dim=1)
    critic_output = self.neural_network(critic_observation)
    squeezed_critic_output = torch.squeeze(critic_output, dim=1)
    return squeezed_critic_output

#========================== DDPG RL Agent ========================#
# Utility function for soft updating of the target estimation neural networks
# soft update algorithm, following original DDPG paper
@torch.no_grad()
def soft_update_target_estimation_model(acting_model, target_estimation_model, tau):
  for acting_parameter, target_parameter in zip(acting_model.parameters(), target_estimation_model.parameters()):
    target_parameter.copy_(acting_parameter * tau + target_parameter * (1. - tau))

# Definition of RL Agent
class DDPG:
  def __init__(self,\
               agent_id_str,\
               # hyperparameters
               actor_lr_initial, critic_lr_initial, soft_update_tau_initial, rl_discount_factor_initial,\
               # ANN size
               forecast_length, additional_info_length, hidden_size, action_length,\
               # device for storing neural network parameters and computing
               device=torch.device('cpu')):
    
    self.device = device

    # parameter update counter starts from 0. every time learn_from() is run, this counter is incremented by 1. 
    self.parameter_update_counter = 0

    # actor
    self.actor = Actor(forecast_length, additional_info_length, hidden_size, action_length, self.device)
    self.target_estimating_actor = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.neural_network.parameters(), lr=actor_lr_initial)

    # critic
    self.critic = Critic(actor_observation_size=self.actor.observation_size,\
                               actor_action_length=self.actor.inference_length,\
                               hidden_size=hidden_size,\
                               device=self.device)
    self.target_estimating_critic = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.neural_network.parameters(), lr=critic_lr_initial)

    # used by save_agent() method. These are saved .pt filenames.
    self.roles_dict = {
        'actor': self.actor,
        'target_estimating_actor': self.target_estimating_actor,
        'critic': self.critic,
        'target_estimating_critic': self.target_estimating_critic
    }

    # used when updating target network's parameters using 'soft update'. The operation is similar to exponential moving average.
    self.tau = soft_update_tau_initial

    # for y = r + discount*Q'(s(i+1),a'(i+1)). The dash ' here means target estimation, not derivatives.
    self.discount_factor = rl_discount_factor_initial

    # if the dir for this agent name does not exist yet, create a directory for saving the critic and actor neural network parameters
    self.agent_dir = os.path.join(config.dir_names['training_output'], agent_id_str)
    if not os.path.exists(self.agent_dir):
      os.makedirs(self.agent_dir)
      
  def print_network_structure(self):
    print('\n****************************** Neural Network Structure *************************************')
    print('Actor:')
    print(self.actor.neural_network)
    print('Critic:')
    print(self.critic.neural_network)
    print('Trainable parameters in DDPG actor : ', sum(parameters.nelement() for parameters in self.actor.neural_network.parameters()))
    print('Trainable parameters in DDPG critic: ', sum(parameters.nelement() for parameters in self.critic.neural_network.parameters()))

  def save_agent(self, path=None):
    target_dir = self.agent_dir if path is None else path
    # create the target_dir if it does not exist
    if not os.path.exists(target_dir):
      os.makedirs(target_dir)
    for role_name_in_text, role in self.roles_dict.items():
      torch.save(
        role.neural_network.state_dict(),
        os.path.join(target_dir, f'{role_name_in_text}.pt')
      )
    return f"\nAgent parameters are saved at {target_dir}"

  @classmethod
  def load_model(cls, run_id):
    ddpg_agent = cls(agent_id_str=f'ddpg_{run_id}',
                 actor_lr_initial=config.learning_rate,
                critic_lr_initial=config.learning_rate,
          soft_update_tau_initial=config.soft_update_tau,
       rl_discount_factor_initial=config.discount_factor,
                  forecast_length=config.forecast_length_per_observation,
           additional_info_length=config.soc_length,
                      hidden_size=config.hidden_size,
                    action_length=config.actor_inference_length,
                           device=torch.device('cpu'))

    for role_name_in_text, role in ddpg_agent.roles_dict.items():
      role.neural_network.load_state_dict(
          torch.load(os.path.join(ddpg_agent.agent_dir, f'{role_name_in_text}.pt'),  map_location=torch.device('cpu'))
      )
    return ddpg_agent

  # def load_agent(self, model_path, parameter_update_counter:int):
  #   '''
  #   old method for loading parameters
  #   '''
  #   # read stored parameters from the directory and load them into the neural network
  #   for role_name_in_text, role in self.roles_dict.items():
  #     role.neural_network.load_state_dict(
  #         torch.load(os.path.join(model_path, str(parameter_update_counter), f'{role_name_in_text}.pt'), map_location=torch.device('cpu'))
  #     )
  #   # set self.parameter_update_counter to the value of the parameter_update_counter
  #   self.parameter_update_counter = parameter_update_counter
  #   print("Parameter Loaded.")
  #   print(f"{self.parameter_update_counter=}")
  
  # action = ref_to_RL_agent.predict(observation_T_tensor, additive_noise_scale)
  def predict(self, observation, noise_scale=None):
    '''
    wrapping actor's forward method to optionally add noise to the action. 
    '''
    actor_output = self.actor.forward(observation)
    if noise_scale is not None:
      noise_scalar      = config.exploration_noise_scale_decay ** self.parameter_update_counter * noise_scale
      exploration_noise = torch.randn_like(actor_output, device=self.device) * noise_scalar
      action            = torch.add(actor_output, exploration_noise)
    else: 
      action = actor_output
    return action

  def learn_from(self, ref_to_batch, loop_counter: int, critic_updates_per_actor_update: int):
    self.parameter_update_counter += 1
    # calculate "better" output values of Q-function
    with torch.no_grad():
      # target estimation does not use backward to update their parameters
      # forward passing through the target estimation networks: actor and then critic 
      action_by_target_estimating_actor = self.target_estimating_actor.forward(ref_to_batch.next_observation)
      y_batch = ref_to_batch.reward.squeeze() + self.discount_factor * self.target_estimating_critic.forward(ref_to_batch.next_observation, action_by_target_estimating_actor)
    
    # calculate Q-function output values using the critic nework that uses backward for updating parameters
    Q_batch = self.critic.forward(ref_to_batch.observation, ref_to_batch.action)

    # calculate mse loss for Q vs target Q
    mse_loss = nn.MSELoss()
    Q_estimation_loss = mse_loss(Q_batch, y_batch) # the Q_estimation_loss is the loss between the output of the current Q network and the Q value that is calculated with slightly more fact, which is the reward(T).

    # update Q network
    self.critic_optimizer.zero_grad()
    Q_estimation_loss.backward()
    self.critic_optimizer.step()

    # update target networks
    with torch.no_grad():
      soft_update_target_estimation_model(self.critic.neural_network, self.target_estimating_critic.neural_network, self.tau)
    
    if loop_counter % critic_updates_per_actor_update == 0:
      # update actor network
      policy_loss = -self.critic.forward(ref_to_batch.observation, self.actor.forward(ref_to_batch.observation)).mean()
      self.actor_optimizer.zero_grad()
      policy_loss.backward()
      self.actor_optimizer.step()
      wandb.log({"Q_estimation_loss": Q_estimation_loss.item(),\
                 "policy_loss": policy_loss.item()},\
                 step=self.parameter_update_counter)
      with torch.no_grad():
        soft_update_target_estimation_model(self.actor.neural_network, self.target_estimating_actor.neural_network, self.tau)

#============================= Reward Function ========================================#
# reward function uses this utility: mae loss implementation for tensor
@torch.no_grad()
def mae_loss_function(input: torch.Tensor, target: torch.Tensor):
  abs_diff = torch.abs(input - target)
  mae = torch.mean(abs_diff, dim=-1, keepdim=True)
  return mae
@torch.no_grad()
def abs_error_function(input: torch.Tensor, target: torch.Tensor):
  abs_diff = torch.abs(input - target)
  return abs_diff

def mse_loss_np(input: np.ndarray, target: np.ndarray):
  mse = np.mean((input - target)**2)
  return mse

def rmse_loss_np(input: np.ndarray, target: np.ndarray):
  mse = mse_loss_np(input, target)
  rmse = np.sqrt(mse)
  return rmse

# reward function that evaluates the goodness of an action (control plan)
@torch.no_grad()
def reward_function(control_plan_tensor:     torch.Tensor,\
                    battery_power_tensor:    torch.Tensor,\
                    net_output_power_tensor: torch.Tensor,\
                    forecast_power_tensor:   torch.Tensor,\
                    unconstrained_PV_power_output_tensor: torch.Tensor,\
                    reward_selection = 0) -> torch.Tensor:
  
  mae_between_net_power_and_plan = mae_loss_function(net_output_power_tensor, control_plan_tensor)
  abs_error = abs_error_function(net_output_power_tensor, control_plan_tensor)
  plan_is_close_to_net_output = torch.isclose(control_plan_tensor, net_output_power_tensor)
  # mae_between_net_power_and_pv_power = mae_loss_function(net_output_power_tensor, unconstrained_PV_power_output_tensor)
  # rmse = rmse_loss_np(control_plan_tensor.detach().cpu().numpy(), net_output_power_tensor.detach().cpu().numpy())
  
  # flow control that uses one integer to switch between two reward functions. The reward is assigned to reward as return.
  if reward_selection == 0:
    # reward function implementation in DeepComp Solar code: 
    # ```python
    # disp = real - P_c + P_d
    # error = pred - disp
    # error_function = abs(error) + beta_c*P_c + beta_d*P_d
    # reward = -error_function
    # ```
    deep_comp_reward_function = -torch.abs(net_output_power_tensor  - control_plan_tensor) - torch.abs(battery_power_tensor)
    reward_tensor = torch.mean(deep_comp_reward_function, dim=-1, keepdim=True)
  elif reward_selection == 1:
    reward_tensor = torch.sum(plan_is_close_to_net_output.int(), dim=-1, keepdim=True)\
                        -torch.sum(control_plan_tensor < config.zero_power_threshold, dim=-1, keepdim=True)\
                        -torch.sum(control_plan_tensor >=1, dim=-1, keepdim=True)\
                        -torch.sum(net_output_power_tensor < config.zero_power_threshold, dim=-1, keepdim=True) 

  elif reward_selection == 2:
    reward_tensor = torch.sum(plan_is_close_to_net_output.int(), dim=-1, keepdim=True)\
                        *torch.all(control_plan_tensor != 0, dim=-1, keepdim=True).float()\
                        - mae_between_net_power_and_plan 
                        #- mae_between_net_power_and_pv_power
  elif reward_selection == 3:
    reward_tensor = torch.sum(plan_is_close_to_net_output.int(), dim=-1, keepdim=True)\
                        *torch.all(control_plan_tensor != 0, dim=-1, keepdim=True).float()\
                        - abs_error 
  return reward_tensor

#================= generate Experience for critic training ==============================================#  
@torch.no_grad()
def generate_exp_from_tensor(ref_to_data: torch.Tensor,\
                             ref_to_RL_env: Battery,\
                             ref_to_RL_agent: DDPG,\
                             ref_to_replay_buffer: Replay_Buffer,\
                             current_RL_iteration: int,\
                             exploration_noise_basis: float=0.,\
                             validating_switch: bool=False,\
                             test_flag: bool=False):
  
  number_of_soc_levels_to_consider = config.number_of_soc_levels_for_validating if validating_switch else config.number_of_soc_levels_for_training
  # expand (view) the data tensor to prepare for additive exploration noise and SoC situations.
  unsqueezed_ref_to_data = ref_to_data.unsqueeze(2)
  expanded_unsqueezed_ref_to_data = unsqueezed_ref_to_data.expand(-1,-1,\
                                                                  number_of_soc_levels_to_consider,\
                                                                  -1,-1)
  # dim0=day, dim1=no. of sample in a day, dim2=number of soc levels, dim3=time series index in one sample
  forecasted_pv_power, measured_pv_power = torch.unbind(expanded_unsqueezed_ref_to_data, dim=3)

  # move forecasted_pv_power and measured_pv_power tensors to  the same device as the RL agent for computation
  forecasted_pv_power_tensor_on_device = forecasted_pv_power.to(ref_to_RL_agent.device)
  measured_pv_power_tensor_on_device   = measured_pv_power.to(ref_to_RL_agent.device)
  
  soc_tensor_shape_as_a_list     = list(forecasted_pv_power.shape)
  soc_tensor_shape_as_a_list[-1] = 1
  soc_level_tensor     = torch.linspace(0,100, number_of_soc_levels_to_consider, device=ref_to_RL_agent.device)\
                              .expand(soc_tensor_shape_as_a_list[0], soc_tensor_shape_as_a_list[1], -1)\
                              .unsqueeze(-1) # unsqueeze to make this a 4-dimensional tensor create a new dimension of size one after existing dimensions for concatenating with forecast tensor at dim3
  # make noise for initial SoC 
  # If gen_exp is run for validating, the noise should be zero for every simulation.
  soc_noise          = validating_switch\
                      *torch.randn_like(soc_level_tensor, device=ref_to_RL_agent.device)
  
  initial_soc_tensor = torch.clamp(soc_level_tensor + soc_noise,\
                                   config.soc_lower_limit_percent,\
                                   config.soc_upper_limit_percent)

  # assemble the forecast and SoC tensors alone dim3 to obtain the observation tensor
  observation_T_tensor = torch.concat((forecasted_pv_power_tensor_on_device[::,:-1:,::,::] * config.forecast_scalar,\
                                      initial_soc_tensor[::,:-1:,::,::] * config.soc_scalar\
                                      ),dim=3)
  
  # # assign actor_output using observation_T_tensor as the input to the actor nueral network.
  # actor_output = ref_to_RL_agent.actor.forward(observation_T_tensor)
  # noise_scalar = config.exploration_noise_scale_decay**current_RL_iteration * exploration_noise_basis
  # exploration_noise = torch.randn_like(actor_output, device=ref_to_RL_agent.device) * noise_scalar
  # # apply scaled additive noise to the actor output
  # action_with_exploration_noise = torch.add(actor_output, exploration_noise)
  action = ref_to_RL_agent.predict(observation_T_tensor, exploration_noise_basis)
  # obtaining the control plan by adding the noisy actor output to the forecasted PV power
  control_plan = torch.clamp(torch.add(action, forecasted_pv_power_tensor_on_device[::,:-1:,::,::]), 0, 1)

  # simulate for the net power output and SoC for each control_plan horizon, that is the identical horizon as the PV forecast.
  net_output_power_tensor, battery_power_tensor, battery_soc_tensor = ref_to_RL_env.simulate(control_plan, measured_pv_power_tensor_on_device[::,:-1:,::,::], initial_soc_tensor[::,:-1:,::,::])
  
  # T is current time, "+1" means "+resolution * 1"
  initial_soc_T_plus_one_tensor        = torch.empty_like(initial_soc_tensor[::,:-1:,::,::], dtype=torch.float32, device=ref_to_replay_buffer.batch_device)
  initial_soc_T_plus_one_tensor[...,0] = battery_soc_tensor[...,config.next_forecast_produced_after_N]
  observation_T_plus_one_tensor        = torch.concat((forecasted_pv_power_tensor_on_device[::,1::,::,::] * config.forecast_scalar, initial_soc_T_plus_one_tensor * config.soc_scalar), dim=3)

  # Performance Monitoring
  rmse = rmse_loss_np(control_plan.detach().cpu().numpy(), net_output_power_tensor.detach().cpu().numpy())
  wandb.log({"RMSE_in_exp_gen": rmse}, step=1 if test_flag else ref_to_RL_agent.parameter_update_counter)
  # Control Plan Performance Indicator 1: how many points in the control plan became reality, that is the net power output.
  # count the number of control plans muted by forecasted_pv_power_T_tensor < config.zero_power_threshold
  muted_points_count  = torch.sum(forecasted_pv_power_tensor_on_device[::,::,0,0] < config.zero_power_threshold)
  # count the number of perfect planning points in all the control plans produced in this validation (muted points deducted)
  perfect_plan_count = torch.isclose(control_plan, net_output_power_tensor).sum() - muted_points_count
  # count the number of control plans made (muted points deducted)
  control_plan_count = control_plan.numel() - muted_points_count 
  perfect_plan_percentage = perfect_plan_count / control_plan_count * 100
  
  wandb.log({"Perfect Plan Percentage": perfect_plan_percentage}, step=1 if test_flag else ref_to_RL_agent.parameter_update_counter)
  # calculate reward for each control plan using reward_function
  reward_tensor = reward_function(control_plan,\
                                  battery_power_tensor,\
                                  net_output_power_tensor,\
                                  forecasted_pv_power_tensor_on_device[::,:-1:,::,::],\
                                  measured_pv_power_tensor_on_device[::,:-1:,::,::],\
                                  reward_selection = config.reward_function_selector)
  mean_reward = torch.mean(reward_tensor, dim=None)
  wandb.log({'mean_reward_in_exp_gen': mean_reward.item()}, step=1 if test_flag else ref_to_RL_agent.parameter_update_counter)

  # memorize the experience in the replay buffer
  ref_to_replay_buffer.memorize(observation_T_tensor, action, reward_tensor, observation_T_plus_one_tensor)
  #  save agent parameters if the agent shows good performance
  #     if perfect_plan_percentage > config.save_agent_threshold_percentage:
  #       logging.info(f"\nValidation shows Perfect Plan Percentage is {perfect_plan_percentage}% (>{config.save_agent_threshold_percentage}%).")
  #       logging.info("\nSaving this agent ...")
  #       save_info = ref_to_RL_agent.save_agent(path)
  #       config.save_info_buffer.append(save_info)
  #       logging.info("\nAgent parameter is saved.")

def default_tensor_operation_device():
  # Use CUDA (NVIDIA GPU) or MPS (Apple Silicon) for tensor operations (backpropagation) if available.
  if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('*Using MPS')

  elif torch.cuda.is_available():
      device = torch.device('cuda')
      print('*Using CUDA')
  else:
    device = torch.device('cpu')
    print('*No GPU, using CPU and RAM')
  return device
