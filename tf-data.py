import os
from slippi import Game

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

def _float_feature(values):
  for v in values:
    assert(isinstance(v, float))
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_feature(values):
  for v in values:
    assert(isinstance(v, int))
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

TYPE_FEATURE_MAP = {
  float: _float_feature,
  int: _int64_feature,
}

player_paths = {
  'pre': {
    'buttons': {
      'physical': int,
    },
    'joystick': {
      'x': float,
      'y': float,
    },
    'cstick': {
      'x': float,
      'y': float,
    }
  },
  'post': {
    'position': {
      'x': float,
      'y': float,
    },
    'character': int,
    'state': int,
    'state_age': float,
    'damage': float,
    'shield': float,
  },
}

game_paths = {i: player_paths for i in range(2)}

_omit = ['physical', 'post']
_rename = {
  'pre': 'controller',
}

def adjust_path(path):
  return [_rename.get(name, name) for name in path if name not in _omit]

def write_flat(path, feature, d, adjust=True):
  path = path.split('/')
  if adjust: path = adjust_path(path)
  flat_key = '.'.join(path)
  d[flat_key] = feature

def append_data(src, dest):
  for key, buf in dest.items():
    if isinstance(key, int):
      value = src[key]
    else:
      value = getattr(src, key)
    if isinstance(buf, list):
      buf.append(value)
    else:
      append_data(value, buf)

def to_tf_example(game):
  if game.start.is_teams:
    return None

  player_ports = [i for i, player in enumerate(game.start.players) if player is not None]
  game_buffer = nest.map_structure(lambda _: [], game_paths)

  for frame in game.frames[:2]:
    for i, p in enumerate(player_ports):
      data = frame.ports[p].leader
      append_data(data, game_buffer[i])

  _to_feature = lambda t, l: TYPE_FEATURE_MAP[t](l)

  game_features = nest.map_structure_up_to(game_paths, _to_feature, game_paths, game_buffer)
  flat_features = {}
  nest.map_structure_with_paths(write_flat, game_features, d=flat_features)

  return tf.train.Example(features=tf.train.Features(feature=flat_features))

replay_files = []

for dirpath, _, filenames in os.walk('replays/Gang-Steals/stream'):
  for fname in filenames:
    replay_files.append(os.path.join(dirpath, fname))

test_game = Game(replay_files[100])

print(len(test_game.frames))

test_example = to_tf_example(test_game)
