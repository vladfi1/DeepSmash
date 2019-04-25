import functools
import pickle
import random

import numpy as np
from tensorflow.python.util import nest
import gym
from ray import rllib

from dsmash import action
from dsmash.rllib import ssbm_spaces
from dsmash.env.reward import rewards_np

@functools.lru_cache()
def get_data(data_path):
  with open(data_path, 'rb') as f:
    return pickle.load(f)


class GameReader:
  def __init__(self, data_path):
    self._games = get_data(data_path)
    self._game = None

  def reset(self):
    self._game = random.choice(self._games)
    self._length = len(self._game.state.stage)
    self._frame = 0
    self._rewards = rewards_np(self._game.state, get=getattr)

  def _get_frame(self, xs):
    return xs[self._frame]

  def poll(self):
    if not self._game:
      self.reset()
    
    state_action = nest.map_structure(self._get_frame, self._game)
    reward = self._rewards[self._frame-1] if self._frame > 0 else 0.

    self._frame += 1
    done = self._frame == self._length
    if done:
      self._game = None

    return state_action, done, reward


class ImitationEnv(rllib.env.BaseEnv):
  def __init__(self, config):
    self._config = config
    self._data_path = config["data_path"]
    self._num_parallel = config.get("num_parallel", 1)
    print("NUM_PARALLEL", self._num_parallel)
    self._readers = [GameReader(self._data_path) for _ in range(self._num_parallel)]
    self._flat_obs = config.get("flat_obs", False)
    
    if self._flat_obs:
      self.observation_space = gym.spaces.Box(
        low=-10, high=10,
        shape=(ssbm_spaces.slippi_game_conv.flat_size,),
        dtype=np.float32)
      self._conv = ssbm_spaces.slippi_game_conv.make_flat
    else:
      self.observation_space = ssbm_spaces.slippi_game_conv.space
      self._conv = ssbm_spaces.slippi_game_conv

    self.action_space = action.to_multidiscrete(
        action.repeated_simple_controller_config)

  def send_actions(self, actions):
    pass

  def poll(self):
    state_actions, dones, rewards = zip(*[r.poll() for r in self._readers])
    
    obs = {i: {0: self._conv(sa.state)} for i, sa in enumerate(state_actions)}
    dones = {i: {"__all__": done} for i, done in enumerate(dones)}
    rewards = {i: {0: r} for i, r in enumerate(rewards)}
    infos = {i: {} for i in range(self._num_parallel)}
    off_policy_actions = {
        i: {0: nest.flatten(sa.action)}
        for i, sa in enumerate(state_actions)}
    
    return obs, rewards, dones, infos, off_policy_actions

  def try_reset(self, env_id):
    return None


if __name__ == "__main__":
  config = dict(data_path='il-data/Gang-Steals/stream_compressed.pkl')
  env = ImitationEnv(config)
  env.poll()

