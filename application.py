import os
import logging
import time
import pprint
import argparse

import torch
import gymnasium as gym
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb

import dfc_gymnasium
import definitions
import config
import helper_fns as hlp

def train_ddpg(dataset: str, bcap=None, agent_properties=None, sweep_config=None):
  # Logging setup
  logging.basicConfig(filename='logging.log', encoding='utf-8', filemode='w', level=logging.INFO)
  if bcap is None:
    battery_capacity_pus = config.battery_capacity_pus
  else:
    battery_capacity_pus = hlp.puh_to_pus(bcap)

  if sweep_config is not None:
    max_RL_epoch = sweep_config.max_RL_epoch
    replay_buffer_capacity = sweep_config.replay_buffer_capacity
    max_batch_size = sweep_config.max_batch_size
    critic_updates_per_RL_epoch = sweep_config.critic_updates_per_RL_epoch
    critic_updates_per_actor_update = sweep_config.critic_updates_per_actor_update
    actor_learning_rate = sweep_config.actor_learning_rate
    critic_learning_rate = sweep_config.critic_learning_rate
  else:
    max_RL_epoch = config.max_RL_epoch
    replay_buffer_capacity = config.replay_buffer_capacity
    max_batch_size = config.max_batch_size
    critic_updates_per_RL_epoch = config.critic_updates_per_RL_epoch
    critic_updates_per_actor_update = config.critic_updates_per_actor_update
    actor_learning_rate = config.learning_rate
    critic_learning_rate = config.learning_rate

  device       = definitions.default_tensor_operation_device()
  data_path    = os.path.join(config.data_register[dataset]['path'])
  data_feature = config.data_register[dataset]['feature']
  data_buffer  = definitions.Data_Buffer(device=device)
  data_buffer.read_data(data_path, *data_feature['file_structure'])
  env = definitions.Battery(battery_capacity_pus,
                            config.soc_upper_limit_percent,
                            config.soc_lower_limit_percent,
                            config.e_to_fcharge_sec,
                            config.e_to_fcharge_sec,
                            charging_efficiency=config.charging_efficiency,
                          discharging_efficiency=config.discharg_efficiency,
                              time_per_step_sec=data_feature['temporal_resolution'],
                                          device=device)

  # split data_buffer into training and validating data sets
  training_tensor, _ = data_buffer.unfold_according_to_size(config.forecast_length_per_observation)

  # initialize replay buffer
  replay_buffer = definitions.Replay_Buffer(training_tensor.shape,
                                            replay_buffer_capacity,
                                            batch_device=device,
                                            storage_device=device)

  ddpg_agent  = definitions.DDPG(agent_id_str=f"ddpg_{agent_properties['id']}",
                             actor_lr_initial=actor_learning_rate,
                            critic_lr_initial=critic_learning_rate,
                      soft_update_tau_initial=config.soft_update_tau,
                   rl_discount_factor_initial=config.discount_factor,
                              forecast_length=config.forecast_length_per_observation,
                       additional_info_length=config.soc_length,
                                  hidden_size=config.hidden_size,
                                action_length=config.actor_inference_length,
                                       device=device)

  print(f'\n------------------------ train_ddpg(): Start Training with dataset {dataset} ----------------------------- ')
  print(f'train_ddpg(): {len(training_tensor)} days of data loaded for training.')
  print(f'train_ddpg(): {training_tensor.shape=} {training_tensor.device=}')
  for i in tqdm(range(max_RL_epoch), desc='RL Epoch', position=0):
    # generate experience and store them in the replay buffer, using the noise amplitude given in the list
    # for explore_noise_basis in [0, 0.1, 0.5, 0.9]:
    for explore_noise_basis in [0, 0.1]:
      definitions.generate_exp_from_tensor(training_tensor, env, ddpg_agent, replay_buffer, i, exploration_noise_basis=explore_noise_basis, validating_switch=False)

    # update network parameters
    batch_size = min(max_batch_size, replay_buffer.len()//5)
    wandb.log({'replay_buffer_length': replay_buffer.len()})
    for j in range(critic_updates_per_RL_epoch):  
      batch = replay_buffer.randomly_sample_a_batch(batch_size)
      ddpg_agent.learn_from(batch,loop_counter=j, critic_updates_per_actor_update=critic_updates_per_actor_update)
    # validate the result of the last network parameter update  
    if i == max_RL_epoch - 1:
      definitions.generate_exp_from_tensor(training_tensor, env, ddpg_agent, replay_buffer, i, exploration_noise_basis=0, validating_switch=False)

  save_info = ddpg_agent.save_agent()
  print('\n----------------------------------- Training Finished ------------------------------------------- ')
  
#  config.save_info_buffer.append(save_info)
#  # Print saved info
#  for text in config.save_info_buffer:
#    print(text)
  
def train_ddpg_tracked_by_wandb(dataset, bcap):
  # wandb setup

  wandb_run = wandb.init(project='R2022_2_DFC', 
                        job_type='train',
                        config={
                           'bcap_puh': bcap,
                           'replay_buffer_capacity': config.replay_buffer_capacity,
                           'max_batch_size': config.max_batch_size,
                           'max_RL_epoch': config.max_RL_epoch,
                           'critic_updates_per_RL_epoch': config.critic_updates_per_RL_epoch,
                           'critic_updates_per_actor_update': config.critic_updates_per_actor_update,
                           'reward_function_selector': config.reward_function_selector,
                           'learning_rate': config.learning_rate,
                           'discount_factor': config.discount_factor,
                           'soft_update_tau': config.soft_update_tau,},
                        group=f'{dataset}-0.33E',
                        tags=['DDPG', 'preproduction'])
  train_ddpg(dataset=dataset, bcap=bcap, agent_properties={'id': wandb_run.id})
  wandb_run.finish()
  return wandb_run.id

def sweep_train(sweep_config=None):
  with wandb.init(config=sweep_config) as run:
    sweep_config = wandb.config
    train_ddpg(dataset=config.datat, agent_properties={'id': run.id}, sweep_config=sweep_config)


def analyze_evaluation(filepath: str, spotlight: bool, atol_pu: float):
  df = pd.read_csv(filepath)
  if spotlight:
    df = df[df['Pnet'] != df['Pdfc']]
  Pm   = df['env_state_pv_potential'].to_numpy()
  Pnet = df['Pnet'].to_numpy()
  Pdfc = df['Pdfc'].to_numpy()
  actual_c = df['actual_cr'].to_numpy()

  mean_curtailment_ratio = hlp.mean_curtailment_ratio(actual_c)
  mean_curtailment = hlp.mean_curtailment(Pm, actual_c)
  perfect_rate = hlp.perfect_plan_rate(Pnet, Pdfc, atol_pu)
  rmse = hlp.rmse_loss_np(Pnet, Pdfc)
  mae = hlp.mae_loss_np(Pnet, Pdfc)
  mape = hlp.mape_loss_np(Pnet, Pdfc)

          
  metric_dict = {
    "filename": os.path.basename(filepath),
    "mean_curtailment_ratio": mean_curtailment_ratio,
    "mean_curtailment": mean_curtailment,
    "perfect_rate": perfect_rate,
    "rmse": rmse,
    "mae": mae,
    "mape": mape, # warning: adding this column will corrupt existing evaluation_journal.csv as it did not journalize mape.
    "spotlight": spotlight,
  }
  return metric_dict

def journalize_evaluation(csv_path: str, fixed_c, run_id: str, dataset: str, bess_properties: dict):
  journal_filename = f"evaluation_journal_{dataset}.csv"
  if dataset.startswith('Elia'):
    atol_pu = 1e-08
  else:
    atol_pu = config.data_register[dataset]['atol'] / config.data_register[dataset]['power_base']
# atol_pu = config.data_register[dataset]['atol'] / config.data_register[dataset]['power_base']
  print(f"journalize_evaluation(): {dataset=}, {atol_pu=}")
  info_dict   = {
    "dataset": dataset,
    "actor"  : run_id,
    "fixed_c": fixed_c,
    "BESS_capacity": bess_properties["energy_capacity_puh"],
    "BESS_charg_lim": bess_properties["power_boundary_Erate"].limsup,
    "BESS_disch_lim": bess_properties["power_boundary_Erate"].liminf,
  }
  info_dict.update(analyze_evaluation(csv_path, spotlight=False, atol_pu=atol_pu))
  info_df = pd.DataFrame([info_dict])
  print("journalize_evaluation():\n", info_df)
  info_df.to_csv(journal_filename, mode='a', header= not os.path.exists(journal_filename), index=False)

def extract_from_obs(obs):
  pf = obs['pv_forecast'] 
  soc = obs['initial_soc_divided_by_100'] * config.soc_scalar *100
  return np.concatenate([pf, soc])

def eval_ddpg(run_id: str, dataset: str, bess_properties: dict):
  start_time = time.time()
  print('----------------------------------- Evaluation Starts -------------------------------------------')
  logging.basicConfig(filename='logging.log', encoding='utf-8', filemode='w', level=logging.INFO)
  device       = definitions.default_tensor_operation_device()
  data_path    = os.path.join(config.data_register[dataset]['path'])
  data_feature = config.data_register[dataset]['feature']
  ddpg_agent   = definitions.DDPG.load_model(run_id)

  # declare data buffer that reads testing data from data_testing folder.
  data_buffer = definitions.Data_Buffer()
  data_buffer.read_data_from(data_path, *data_feature['file_structure'])
  evaluation_set = data_buffer.prepare_data_np()

  # prepare evaluation output directory
  csv_path = os.path.join(config.dir_names['evaluation_output'], f"{dataset}", f"{run_id}_{dataset}.csv")
  if not os.path.exists(os.path.dirname(csv_path)):
    os.makedirs(os.path.dirname(csv_path))
    
  # create environment for evaluation
  env = gym.make(config.training_config["env_id"],
    forecast_scada_timeseries = evaluation_set,
    bess_properties = bess_properties,
    sec_per_step = data_feature["temporal_resolution"],
    soc_levels = 1,
    render_mode = "evaluation",
    csv_path = csv_path,
    )

  #*** Evaluation Loop ***#
  print(f'eval_ddpg(): evaluating model {run_id}\n{dataset=}')
  pprint.pprint(bess_properties)
  obs, info = env.reset()
  obs_np = extract_from_obs(obs)
  max_steps =  info["max_steps"]
  for i in tqdm(range(max_steps)):
    action = ddpg_agent.actor.forward(torch.from_numpy(obs_np)).detach().cpu().numpy()#, deterministic=True)
    # clip the dfc (=a + pf) with (0,1); append zero for 0 curtailment
    dfc_action = np.concatenate([np.clip(action + obs['pv_forecast'],0,1), 
                                 np.array([0], dtype=np.float32)])  
    obs, _, terminated, _, info = env.step(dfc_action)
    if terminated and not info["last_cluster"]:
      obs, info = env.reset()
    obs_np = extract_from_obs(obs)

  # close the environment 
  env.close()

  end_time = time.time()
  simulation_time = end_time - start_time
  print(f"eval_ddpg(): {max_steps} steps finished in {simulation_time: .2f} seconds")
  journalize_evaluation(csv_path, 0, run_id, dataset, bess_properties)
 
  # visualize the evaluation csv
  target_dir   = os.path.dirname(csv_path)
  figure_title = f"{csv_path}"
  hlp.visualize_pnet_pdfc(csv_path, target_dir, figure_title, save=True, show=False, show_datetime=False)

  return csv_path
def eval_control_i(dataset):
  print(f'----------------------------------- Analyzing {dataset} -------------------------------------------')
  logging.basicConfig(filename='logging.log', encoding='utf-8', filemode='w', level=logging.INFO)
  device       = definitions.default_tensor_operation_device()
  data_path    = os.path.join(config.data_register[dataset]['path'])
  data_feature = config.data_register[dataset]['feature']
  if dataset.startswith('Elia'):
    atol_pu = 1e-08
  else:
    atol_pu = config.data_register[dataset]['atol'] / config.data_register[dataset]['power_base']

  # declare data buffer that reads testing data from data_testing folder.
  data_buffer = definitions.Data_Buffer()
  data_buffer.read_data_from(data_path, *data_feature["file_structure"])
  evaluation_set = data_buffer.prepare_data_pd()
  df = evaluation_set[evaluation_set['Pm'] > 0]
  Pm = df['Pm']
  Pf = df['Pf']
  rmse = hlp.rmse_loss_np(Pm, Pf)
  mape = hlp.mape_loss_np(Pm, Pf)
  perfect_rate = hlp.perfect_plan_rate(Pm, Pf, atol_pu)
  print(dataset, '\t', f'{rmse=}', f'{mape=}', f'{perfect_rate=}')

def eval_control_ii(run_id: str, dataset: str, bess_properties: dict):
  start_time = time.time()
  print('----------------------------------- Evaluation Starts -------------------------------------------')
  logging.basicConfig(filename='logging.log', encoding='utf-8', filemode='w', level=logging.INFO)
  device       = definitions.default_tensor_operation_device()
  data_path    = os.path.join(config.data_register[dataset]['path'])
  data_feature = config.data_register[dataset]['feature']

  # declare data buffer that reads testing data from data_testing folder.
  data_buffer = definitions.Data_Buffer()
  data_buffer.read_data_from(data_path, *data_feature["file_structure"])
  evaluation_set = data_buffer.prepare_data_np()

  # prepare evaluation output directory
  csv_path = os.path.join(config.dir_names['evaluation_output'], f"{dataset}", f"{run_id}_{dataset}.csv")
  if not os.path.exists(os.path.dirname(csv_path)):
    os.makedirs(os.path.dirname(csv_path))
    
  # create environment for evaluation
  env = gym.make(config.training_config["env_id"],
    forecast_scada_timeseries = evaluation_set,
    bess_properties = bess_properties,
    sec_per_step = data_feature["temporal_resolution"],
    soc_levels = 1,
    render_mode = "evaluation",
    csv_path = csv_path,
    )

  #*** Evaluation Loop ***#
  print(f'eval_ddpg(): evaluating model {run_id}\n{dataset=}')
  pprint.pprint(bess_properties)
  obs, info = env.reset()
  obs_np = extract_from_obs(obs)
  max_steps =  info["max_steps"]
  for i in tqdm(range(max_steps)):
    action = 0
    # clip the dfc (=a + pf) with (0,1); append zero for 0 curtailment
    dfc_action = np.concatenate([np.clip(action + obs['pv_forecast'],0,1), 
                                 np.array([0], dtype=np.float32)])  
    obs, _, terminated, _, info = env.step(dfc_action)
    if terminated and not info["last_cluster"]:
      obs, info = env.reset()
    obs_np = extract_from_obs(obs)

  # close the environment 
  env.close()

  end_time = time.time()
  simulation_time = end_time - start_time
  print(f"eval_ddpg(): {max_steps} steps finished in {simulation_time: .2f} seconds")
  journalize_evaluation(csv_path, 0, run_id, dataset, bess_properties)
 
  # visualize the evaluation csv
  target_dir   = os.path.dirname(csv_path)
  figure_title = f"{csv_path}"
  hlp.visualize_pnet_pdfc(csv_path, target_dir, figure_title, save=True, show=False, show_datetime=False)

  return csv_path


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train DFC agent.')
  parser.add_argument("--datat",    type=str, help="dataset for training")
  parser.add_argument("--datae",    type=str, help="dataset for evaluation")
  parser.add_argument("--bcap",    type=float, help="Battery capacity")
  args, _ = parser.parse_known_args()
  wandb_run_id = train_ddpg_tracked_by_wandb(bcap=args.bcap, dataset=args.datat)
  bess_properties = config.bess_properties
  bess_properties["energy_capacity_puh"] = args.bcap
  csv_path = eval_ddpg(run_id=wandb_run_id, dataset=args.datae, bess_properties=bess_properties)
