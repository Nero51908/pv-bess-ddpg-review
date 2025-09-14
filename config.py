import os
# local modules
import helper_fns as hlp
from helper_fns import StateSpaceLimits

dir_names = {
  'training_output'   : 'models',
  'evaluation_output' : 'evaluation',
  'visualization'     : 'evaluation',
}
#############################
# Training Settings (ddpg)
#############################
training_config = {
"total_steps": 500_000,
"env_id": "dfc_gymnasium/UtilityScalePVBESS-v0-nocurtailment",
}

################################
#  properties of input data    #
################################
forecast_length_per_observation = 1 # points of power data in each forecast observable by the actor
next_forecast_produced_after_N = 1 # forecast lead time expressed in discrete time steps. The time an observation is made and control plan is produced is called T=n; the time the next control plan is made is T+1 which is n+N
soc_length = 1 # the SoC of the battery is represented by a scalar, that is 1 number

#############################
# Input Data Specifications
#############################
# just a note of supported input data formats readable by the DataBuffer.read_data_from()
supported_data_formats = ['NEM_csv', 'Elia_xls', 'DeepComp_csv']
interesting_col_NEM = ['DATETIME', 'FORECAST_POE50', 'SCADAVALUE']
dataFeature_NEM  = {'file_structure': (0,288,interesting_col_NEM,'NEM_csv'),
                    'temporal_resolution': 5*60., # 5 minutes expressed in seconds
                  }

# data from Elia (day-ahead forecast, 15 min dispatch interval, 96 points per 24 hours)
interesting_col_Elia_dah = ['DateTime', 'Day-Ahead forecast [MW]', 'Corrected Upscaled Measurement [MW]']
dataFeature_Elia_dah = {'file_structure': (3,96,interesting_col_Elia_dah,'Elia_xls'),
                        'temporal_resolution': 15*60., # 15 minutes expressed in seconds
                      } 
dataFeature_Elia_1h_dah = {'file_structure': (0,24,interesting_col_Elia_dah,'Elia_csv'),
                        'temporal_resolution': 60*60., # 60 minutes expressed in seconds
                      }


# data from Elia (hour-ahead forecast, 15 min dispatch interval, 96 points per 24 hours)
interesting_col_Elia_hah = ['DateTime', 'Most recent forecast [MW]', 'Corrected Upscaled Measurement [MW]']
dataFeature_Elia_hah = {'file_structure': (3,96,interesting_col_Elia_hah,'Elia_xls'),
                        'temporal_resolution': 15*60., # 15 minutes expressed in seconds
                      } 

# Elia processed to have one-hour resolution
dataFeature_Elia_1h_hah = {'file_structure': (0,24,interesting_col_Elia_hah,'Elia_csv'),
                        'temporal_resolution': 60*60., # 60 minutes expressed in seconds
                      }

# data from DeepComp is Elia data
interesting_col_DeepComp = ['Time', 'Forecast(MW)', 'Power(MW)']
dataFeature_DeepComp = {'file_structure': (0,96,interesting_col_DeepComp, 'DeepComp_csv'),
                        'temporal_resolution': 15*60., # 15 minutes expressed in seconds
                      }


# data register (from which directory data files will be found)
data_register = {
  'Elia_2016-2018'        : {'path': os.path.join('data', 'Elia_2016-2018'),    'feature': dataFeature_Elia_dah},
  'Elia_2016-2018_1h_dah' : {'path': os.path.join('data', 'Elia_2016-2018_1h'), 'feature': dataFeature_Elia_1h_dah},
  'Elia_2016-2018_1h_hah' : {'path': os.path.join('data', 'Elia_2016-2018_1h'), 'feature': dataFeature_Elia_1h_hah},
  'Elia_2016'             : {'path': os.path.join('data', 'Elia_2016'),         'feature': dataFeature_Elia_dah},
  'Elia_2019'             : {'path': os.path.join('data', 'Elia_2019'),         'feature': dataFeature_Elia_dah},
  'Elia_2019_1h_dah'      : {'path': os.path.join('data', 'Elia_2019_1h'),      'feature': dataFeature_Elia_1h_dah},
  'Elia_2019_1h_hah'      : {'path': os.path.join('data', 'Elia_2019_1h'),      'feature': dataFeature_Elia_1h_hah},
  'RUGBYR1_old'    : {'path': os.path.join('data', 'NEMweb', 'RUGBYR1',  '202309_202408'), 'feature': dataFeature_NEM, 'power_base': 65.45, 'atol': 6},
  'RUGBYR1_new'    : {'path': os.path.join('data', 'NEMweb', 'RUGBYR1',  '202408_202505'), 'feature': dataFeature_NEM, 'power_base': 65.45, 'atol': 6},
  'BANN1_old'      : {'path': os.path.join('data', 'NEMweb', 'BANN1',    '202309_202408'), 'feature': dataFeature_NEM, 'power_base': 88.84, 'atol': 6},
  'BANN1_new'      : {'path': os.path.join('data', 'NEMweb', 'BANN1',    '202408_202505'), 'feature': dataFeature_NEM, 'power_base': 88.84, 'atol': 6},
  'EDENVSF1_old'   : {'path': os.path.join('data', 'NEMweb', 'EDENVSF1', '202309_202408'), 'feature': dataFeature_NEM, 'power_base': 147.85, 'atol': 6},
  'EDENVSF1_new'   : {'path': os.path.join('data', 'NEMweb', 'EDENVSF1', '202408_202505'), 'feature': dataFeature_NEM, 'power_base': 147.85, 'atol': 6},
}

# specify the datasets used for training and evaluation
#datat = 'Elia_2016-2018'
#datat = 'Elia_2016-2018_1h_hah'
#datae = 'Elia_2019'
#datat = 'BANN1_old'
#datae = 'BANN1_new'
#datat = 'RUGBYR1_old'
#datae = 'RUGBYR1_new'
#datat = 'EDENVSF1_old'
#datae = 'EDENVSF1_new'


#############################
# BESS Properties
#############################
bess_capacity_puh = 0.5 # should be overwritten by application.py when --bcap is specified
battery_capacity_pus = hlp.puh_to_pus(bess_capacity_puh)
soc_upper_limit_percent = 90.0 # percent
soc_lower_limit_percent = 10.0 # percent

# charging and discharging power limit for the battery is specified based on the time in sec taken to full charge or empty cahrge
charging_power_Erate = 1/3
discharg_power_Erate = 1/3
e_to_fcharge_sec = 1/charging_power_Erate * 3600   # fcharge: full charge
f_to_echarge_sec = 1/discharg_power_Erate * 3600 # echarge: empty charge
charging_efficiency = 0.9 # factor to multiply the input power to the battery for the actual power reflected in the SoC
discharg_efficiency = 0.9 # factor to multiply the output power to the battery for the actualy power experienced by the load

bess_properties = {
  "soc_boundary_percent": StateSpaceLimits(soc_upper_limit_percent, soc_lower_limit_percent),
  "power_boundary_Erate": StateSpaceLimits(charging_power_Erate, (-1)*discharg_power_Erate),
  "energy_capacity_puh":  bess_capacity_puh,
  "charging_efficiency":  charging_efficiency,
  "discharging_efficiency": discharg_efficiency,
}

################################
# hyperparameters for training #
################################
# neural network structure #
# actor nn's input feature length is forecast_length_per_observation + soc_length, this is called actor observation size
# critic nn's input feature length is (forecast_length_per_observation + soc_length) + actor_inference_length
# Then both actor and critic have the same hidden_size
hidden_size = 128
actor_inference_length = 1 # the length of the output of actor's neural network

# Actor network input (observation) needs to be scaled for faster and stable convergence over parameter updates because SoC is a percentage that ranges (0,100), but the forecasted PV power is in pu which ranges (0,1)
forecast_scalar = 1
soc_scalar = 0.005 # scale SoC for actor input

# Training Script Setting #
training_data_portion  = 100 # unit: percentage. Percentage of the data input. E.g. 80 means 80% of the data is used for training and 20% for validating.
replay_buffer_capacity = 6e7 #2e+7  # unit: entries of experience(observation_T, action, reward, observation_T+1)
max_batch_size         = 16384 # unit: entries of experience(observation_T, action, reward, observation_T+1)
save_agent_threshold_percentage = 97# percent. if validation shows P(perfect control) bigger than this percentage, agent.save_agent() will be triggered to save the agent parameters.

# training loop
max_RL_epoch = 25 # how many times the gen_exp() will run for generating data for the replay buffer #150
critic_updates_per_RL_epoch = 500 #1500 for BANN1 #1000 for EDENVSF #1000 is good for Rugbyr1 learn_from() will be called this number of times in each RL iteration 
critic_updates_per_actor_update = 50 #10 for BANN1 # 10 for EDENVSF #10 is good for Rugbyr1 every this number of calls of agent.learn_from() will trigger actor parameter update

# RL reward function
reward_function_selector = 2 # 0: DeepComp's reward function, default value in reward_function() is 0.
                             # 1: my reward function under dev 1
                             # 2: my reward function under dev 2
epoch_reward_normalization = False # True: reward tensor is normalized to have zero mean. False: reward is not normalized.

# assumption
zero_power_threshold = 1e-5 # pu, a value of power smaller than this number is deemed zero in generate_exp_from_day().


learning_rate = 1e-4 # hyper parameter for optimizers
discount_factor = 0.99 # hyper parameter gamma in RL
soft_update_tau = 0.005 # tau is between 0 and 1. This is the hyper parameter tau in orignal DDPG paper. Smaller tau implies softer update to the target parameters. In OpenAI Spinning Up, they used polyak rho=0.995. It is the same as tau=0.005.

# generate experience 
# for exploration during training
number_of_soc_levels_for_training = 50
exploration_noise_scale_decay = 0.998
# for validating
number_of_soc_levels_for_validating = 1 # this should be Constant 1.
initial_soc_for_validating = 50 # percent. this SoC is given to the start of each day in validation.
initial_soc_for_evaluation = initial_soc_for_validating
reset_soc_to_initial_soc_daily_in_validation = True # Deafult is True. This functionality has not been implemented # True: SoC is reset to initial_soc_for_validating at the start of each day in validation. False: SoC is not reset to initial_soc_for_validating at the start of each day in validation.

# Printable information buffer
save_info_buffer = []
