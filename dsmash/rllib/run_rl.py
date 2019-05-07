import argparse
import ray
from ray import tune
from ray.rllib import agents

from dsmash.env.ssbm_env import SSBMEnv
from dsmash.rllib import ssbm_env, rllib_env
from dsmash.rllib import human_trainer

parser = argparse.ArgumentParser()
SSBMEnv.update_parser(parser)
parser.add_argument('--num_workers', type=int)
parser.add_argument('--num_envs_per_worker', type=int)
parser.add_argument('--cluster', action='store_true')
parser.add_argument('--vec_env', action='store_true', help="batch using a single vectorized env")
parser.add_argument('--cpu_affinity', action='store_true')
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
train_batch_size = 128 if args.gpu else 2
num_workers = args.num_workers or 1
num_envs = args.num_envs_per_worker or 1
async_env = True
batch_inference = True  # multiple envs per worker
fc_depth = 2
fc_width = 256
delay = 1
use_test_env = False

batch_inference = async_env and batch_inference
vec_env = args.vec_env and batch_inference
base_env = rllib_env.TestEnv if use_test_env else ssbm_env.MultiSSBMEnv
exp_name = "test" if use_test_env else "ssbm"

top_env = base_env
if async_env:
  top_env = rllib_env.BaseMPEnv if vec_env else rllib_env.MPEnv

tune.run_experiments({
  exp_name: {
    "env": top_env,
    "run": human_trainer.HumanTrainer,
    "checkpoint_freq": 100,
    "config": {
      "env_config": {
        "base_env": base_env,
        
        "ssbm_config": args.__dict__,  # config to pass to env class
        "episode_length": None,
        "num_envs": num_envs if vec_env else 1,
        "delay": delay,
        "flat_obs": True,
        "cpu_affinity": args.cpu_affinity,
        #"profile": True,
        "conv": "slippi",
        "action_mode": "slippi_repeat",

        "step_time_ms": 1,
      },
      "optimizer": {
          "train_batch_size": unroll_length * train_batch_size,
          "replay_buffer_num_slots": 4 * train_batch_size + 1,
          "replay_proportion": 0,
          "learner_queue_size": 16,
      },
      #"sample_async": True,
      "sample_batch_size": unroll_length,
      "horizon": 1200,  # one minute
      "soft_horizon": True,

      "num_gpus": (0.4 if batch_inference else 1) if args.gpu else 0,
      "num_cpus_for_driver": 1,
      "num_workers": num_workers if batch_inference else num_envs,
      "num_gpus_per_worker": 0.1 if (batch_inference and args.gpu) else 0,
      "num_cpus_per_worker": (1+num_envs) if batch_inference else 1,
      "num_envs_per_worker": 1 if vec_env else num_envs,
      # "remote_worker_envs": True,
      "autoregressive": True,
      "residual": True,
      "imitation": False,
      "model": {
        "max_seq_len": unroll_length,
        "use_lstm": True,
        "lstm_use_prev_action_reward": True,
        "fcnet_hiddens": [fc_width] * fc_depth,
      }
    }
  }
})

