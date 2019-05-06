import json
from ray import rllib
from ray.rllib import rollout
from ray.rllib.agents import registry
from dsmash.env.ssbm_env import SSBMEnv
from dsmash.rllib import ssbm_env, model, human_trainer

model.register()
registry.ALGORITHMS["human"] = lambda: human_trainer.HumanTrainer

parser = rollout.create_parser()
SSBMEnv.update_parser(parser)

args = parser.parse_args()

config = {
  "env": ssbm_env.MultiSSBMEnv,
  "env_config": {
    "ssbm_config": args.__dict__.copy(),
    "flat_obs": True,
    "conv": "slippi",
    "action_mode": "slippi_repeat",
  },
  "horizon": 1200,
  "soft_horizon": True,
  "num_workers": 0,
  "autoregressive": True,
  "residual": True,
  "imitation": False,
  "model": {
    "custom_model": "human_action",
    "use_lstm": True,
    "lstm_cell_size": 256,
    "lstm_use_prev_action_reward": True,
  },
}

args.config = config

rollout.run(args, parser, config)


