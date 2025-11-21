import wandb
import application

sweep_config = {
  'method': 'random'
}

metric = {
  'name': 'mean_reward_in_exp_gen',
  'goal': 'maximize'
}

parameters_dict = {
   'max_RL_epoch': {
     'values':[10, 25, 50, 100]
   },
  'replay_buffer_capacity': {
    'values':[8e6, 1e7, 2e7]
  },
   'max_batch_size': {
     'values':[256, 8192, 16384]
   },
  'critic_updates_per_RL_epoch': {
    'values':[10, 500, 1000]
  },
  'critic_updates_per_actor_update': {
    'values':[10, 100, 200]
  },
  'actor_learning_rate': {
    'min': 1e5,
    'max': 2e4
  },
  'critic_learning_rate': {
    'min': 1e5,
    'max': 1e3
  },
}

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='R2022_2_DFC')
wandb.agent(sweep_id, function=application.sweep_train, project='R2022_2_DFC', count=5)
