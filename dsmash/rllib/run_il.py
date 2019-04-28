import argparse
import ray
from ray import tune
from ray.rllib import agents

from dsmash.rllib import imitation_env, imitation_trainer
#from dsmash.rllib import model

#model.register()

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, nargs='+', help='path(s) to pickled slippi data')
parser.add_argument('--num_workers', type=int)
parser.add_argument('--num_envs_per_worker', type=int)
parser.add_argument('--cluster', action='store_true')
parser.add_argument('--vec_env', action='store_true', help="batch using a single vectorized env")
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()


if args.cluster:
  ray.init(
    redis_address="10.0.1.45:6379"
  )
else:
  ray.init(
    redis_max_memory=int(4e9),
  #  object_store_memory=int(4e9),
  )

unroll_length = 60
train_batch_size = 512
fc_depth = 2
fc_width = 256

vec_env = args.vec_env and batch_inference
exp_name = "imitation"

config = {
  "data_path": args.data_path,
  "num_gpus": 1 if args.gpu else 0,
  "num_cpus_for_driver": 2,
  "train_batch_size": unroll_length * train_batch_size,
  "sample_batch_size": unroll_length,
  #"soft_horizon": True,
  "num_workers": 0,
  # "remote_worker_envs": True,
  "model": {
    "is_time_major": True,
    #"max_seq_len": unroll_length,
    "use_lstm": True,
    "lstm_cell_size": 256,
    "lstm_use_prev_action_reward": True,
    "fcnet_hiddens": [fc_width] * fc_depth,
  },
}

tune.run_experiments({
  exp_name: {
    "env": "",
    # "env": imitation_env.ImitationEnv,
    #"run": agents.impala.ImpalaTrainer,
    "run": imitation_trainer.ImitationTrainer,
    #"run": agents.a3c.A3CAgent,
    #"run": agents.a3c.A2CAgent,
    "checkpoint_freq": 100,
    "config": config,
  }},
  resume=args.resume,
  raise_on_failed_trial=True,
)
