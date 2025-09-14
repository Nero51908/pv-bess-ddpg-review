import application
import config

wandb_run_id = application.train_ddpg_tracked_by_wandb(dataset=config.datat)
csv_path = application.eval_ddpg(run_id=wandb_run_id, dataset=config.datae, bess_properties=config.bess_properties)
print('experiment.py:')
print(f'Model {wandb_run_id} is trained and evaluated.')
print(f'Training dataset: \t{config.datat}')
print(f'Evaluation dataset: \t{config.datae}')
print(f'Evaluation is saved as {csv_path}')
