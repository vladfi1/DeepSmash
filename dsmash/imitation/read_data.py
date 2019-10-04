import functools
import pickle
import random

import numpy as np
from tensorflow.python.util import nest

from dsmash import ssbm_actions
from dsmash.env.reward import rewards_np


@functools.lru_cache()
def get_data(data_path):
  with open(data_path, 'rb') as f:
    return pickle.load(f)

@functools.lru_cache()
def get_data_multi(data_paths):
  data = []
  for path in data_paths:
    data.extend(get_data(path))

  total_actions = sum(len(game["state"]["stage"]) for game in data)
  total_frames = sum((game["action"]["repeat"] + 1).sum() for game in data)
  print("Loaded %d games, %d actions, %d frames" %
      (len(data), total_actions, total_frames))

  return data


class GameReader:
  """Reads pickled games written by dsmash.slippi.data."""
  def __init__(self, data_paths):
    data = get_data_multi(tuple(data_paths))
    self._games = data
    self._game = None

  def reset(self):
    self._game = random.choice(self._games)
    self._length = len(self._game["state"]["stage"])
    self._frame = 0

    self._rewards = rewards_np(self._game["state"])
    self._prev_rewards = np.concatenate(([0], self._rewards))

    self._dones = np.zeros((self._length,), dtype=bool)
    self._dones[-1] = True

    self._prev_actions = nest.map_structure(
        lambda xs: np.concatenate(([0], xs[:-1])),
        self._game["action"])

    self._sample_batch = dict(
      obs=self._game["state"],
      actions=self._game["action"],
      rewards=np.concatenate((self._rewards, [0])),
      dones=self._dones,
      prev_actions=self._prev_actions,
      prev_rewards=self._prev_rewards)

  def _get_frame(self, xs):
    return xs[self._frame]

  def poll(self):
    if not self._game:
      self.reset()
    
    state_action = nest.map_structure(self._get_frame, self._game)
    reward = self._prev_rewards[self._frame]

    self._frame += 1
    done = self._frame == self._length
    if done:
      self._game = None

    return state_action, done, reward

  def _get_sample_batch(self, length):
    start_frame = self._frame
    end_frame = start_frame + length
    chop = lambda xs: xs[start_frame:end_frame]
    sample_batch = nest.map_structure(chop, self._sample_batch)
    if end_frame == self._length:
      self.reset()
    else:
      self._frame = end_frame
    return sample_batch
  
  def get_sample_batch(self, size):
    if not self._game:
      self.reset()

    available = self._length - self._frame
    
    if available < size:
      sample1 = self._get_sample_batch(available)
      sample2 = self._get_sample_batch(size - available)
      sample_batch = nest.map_structure(
          lambda *xs: np.concatenate(xs),
          sample1, sample2)
    else:
      sample_batch = self._get_sample_batch(size)
    
    return sample_batch

