import argparse
import config
import application
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate DFC prediction.')
  parser.add_argument('--controli',  action='store_true', help='the evaluation of Pf against Pm')
  parser.add_argument('--controlii', action='store_true', help='Dummy actor responds to obs with zero resulting the evaluation of using Pf as DFC')
  parser.add_argument("--model",    type=str, help="Specify wandb run id of trained model")
  parser.add_argument("--datae",    type=str, help="dataset for evaluation")
  parser.add_argument("--bcap",    type=float, help="Battery capacity")
  args, _ = parser.parse_known_args()

  bess_properties = config.bess_properties
  bess_properties["energy_capacity_puh"] = args.bcap

  if args.controli:
    application.eval_control_i(dataset=args.datae,)
  elif args.controlii:
    application.eval_control_ii(run_id=f"control_II_{bess_properties['energy_capacity_puh']}", dataset=args.datae, bess_properties=bess_properties)
  else:
    run_id = args.model
    print(f'Testing eval_ddpg() with model {run_id} and dataset {config.datae}')
    application.eval_ddpg(run_id=run_id, dataset=config.datae, bess_properties=bess_properties)
