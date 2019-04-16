import argparse
import ray
from ray import tune
from ray.rllib import agents

from dsmash.rllib import imitation_env, imitation_trainer
#from dsmash.rllib import model

#model.register()

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='path to pickled slippi data')
parser.add_argument('--num_workers', type=int)
parser.add_argument('--num_envs_per_worker', type=int)
parser.add_argument('--cluster', action='store_true')
parser.add_argument('--vec_env', action='store_true', help="batch using a single vectorized env")
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()


if args.cluster:
  ray.init(
    redis_address="10.0.1.45:6379"
  )
else:
  ray.init(
    redis_max_memory=int(4e9),
    object_store_memory=int(4e9),
  )

unroll_length = 60
train_batch_size = 128
num_workers = args.num_workers or 1
num_envs = args.num_envs_per_worker or 1
batch_inference = True  # multiple envs per worker
fc_depth = 2
fc_width = 256

vec_env = args.vec_env and batch_inference
exp_name = "imitation"

tune.run_experiments({
  exp_name: {
    "env": imitation_env.ImitationEnv,
    #"run": agents.impala.ImpalaTrainer,
    "run": imitation_trainer.ImitationTrainer,
    #"run": agents.a3c.A3CAgent,
    #"run": agents.a3c.A2CAgent,
    "checkpoint_freq": 100,
    "config": {
      "env_config": {
        "data_path": args.data_path,
        "num_parallel": num_envs if vec_env else 1,
        "flat_obs": True,
        #"profile": True,
      },
      "num_gpus": (0.4 if batch_inference else 1) if args.gpu else 0,
      "num_cpus_for_driver": 1,
      "optimizer": {
          "train_batch_size": unroll_length * train_batch_size,
          "replay_proportion": 0,
          "learner_queue_size": 16,
      },
      #"sample_async": True,
      "sample_batch_size": unroll_length,
      #"soft_horizon": True,
      "num_workers": num_workers if batch_inference else num_envs,
      "num_gpus_per_worker": (0.1 if batch_inference else 0) if args.gpu else 0,
      "num_cpus_per_worker": 1,
      "num_envs_per_worker": 1 if vec_env else num_envs,
      # "remote_worker_envs": True,
      "model": {
        #"max_seq_len": unroll_length,
        "use_lstm": True,
        "lstm_use_prev_action_reward": True,
        "fcnet_hiddens": [fc_width] * fc_depth,
      }
    }
  }
})

