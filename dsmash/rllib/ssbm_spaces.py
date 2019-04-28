from collections import OrderedDict
from functools import partial
import copy
from logging import warning
import math

import numpy as np
import tensorflow as tf
from gym import spaces

class Conv:
  def contains(self, x):
    return True
  
  def make_flat(self, obs):
    array = np.zeros((self.flat_size,))
    self.write(obs, array, 0)
    return array

class BoolConv(Conv):
  flat_size = 2

  def __init__(self, name="BoolConv"):
    self.space = spaces.Discrete(2)
    self.name = name
    self.default_value = 0

  def __call__(self, cbool):
    return int(cbool)
  
  def write(self, b, array, offset):
    array[offset + int(b)] = 1
  
  def embed(self, x):
    return tf.one_hot(x, 2)
  
  def make_ph(self, batch_shape):
    return tf.placeholder(tf.bool, batch_shape, self.name)

def clip(x, min_x, max_x):
  return min(max(x, min_x), max_x)

class RealConv(Conv):
  flat_size = 1

  def __init__(self, source, target, verbose=True, default_value=0., name="RealConv"):
    self.source = source
    self.target = target
    m = (target[1] - target[0]) / (source[1] - source[0])
    b = target[0] - source[0] * m
    self.transform = lambda x: m * x + b
    self.space = spaces.Box(min(target), max(target), (), dtype=np.float32)
    self.verbose = verbose
    self.default_value = np.array(default_value)
    self.name = name
    
  def contains(self, x):
    return self.source[0] <= x and x <= self.source[1]

  def _process(self, x):
    if math.isnan(x):
      warning("NaN value in %s" % self.name)
      return self.default_value
    if not self.contains(x):
      if self.verbose:
        warning("%f out of bounds in real space \"%s\"" % (x, self.name))
      x = clip(x, self.space.low, self.space.high)
    return self.transform(x)

  def __call__(self, x):
    return np.array(self._process(x))

  def write(self, x, array, offset):
    array[offset] = self._process(x)
  
  def embed(self, x):
    guard_nan = tf.where(
        tf.is_nan(x),
        tf.fill(tf.shape(x), tf.constant(self.default_value, dtype=x.dtype)),
        x)
    clipped = tf.clip_by_value(guard_nan, self.space.low, self.space.high)
    transformed = self.transform(clipped)
    return tf.expand_dims(transformed, -1)

  def make_ph(self, batch_shape):
    return tf.placeholder(tf.float32, batch_shape, self.name)

def positive_conv(size, *args, **kwargs):
  return RealConv((0, size), (0, 1), *args, **kwargs)

def symmetric_conv(size, out=1., *args, **kwargs):
  return RealConv((-size, size), (-out, out), *args, **kwargs)

class ExceptionConv(Conv):
  def __init__(self, exceptions, name="ExceptionConv"):
    self.exception_dict = {x: i for i, x in enumerate(exceptions)}
    self.space = spaces.Discrete(len(exceptions)+1)
    self.default_value = len(exceptions)
    self.name = name
    self.flat_size = len(exceptions) + 1
  
  def __call__(self, x):
    if x in self.exception_dict:
      return self.exception_dict[x]
    warning("%s out of bounds in exception space '%s'" % (x, self.name))
    return self.default_value

  def contains(self, x):
    return x in self.exception_dict
    
  def write(self, x, array, offset):
    array[offset + self(x)] = 1

class SumConv(Conv):
  def __init__(self, spec, name="SumConv"):
    self.name = name
    self.convs = [f(name=name + '/' + key) for key, f in spec]
    self.space = spaces.Tuple([conv.space for conv in self.convs])
    self.default_value = tuple(conv.default_value for conv in self.convs)
    self.flat_size = sum(conv.flat_size for conv in self.convs)

  def __call__(self, x):
    return_value = list(self.default_value)
    for i, conv in enumerate(self.convs):
      if conv.contains(x):
        return_value[i] = conv(x)
        return return_value
    
    warning("%s out of bounds in sum space '%s'" % (x, self.name))
    return self.default_value
  
  # this doesn't quite do the same thing as the TupleFlattenProcessor
  def write(self, x, array, offset):
    for conv in self.convs:
      if conv.contains(x):
        conv.write(x, array, offset)
        break
      offset += conv.flat_size

class DiscreteConv(Conv):

  def __init__(self, size, name="DiscreteConv"):
    self.size = size
    self.default_value = size
    self.space = spaces.Discrete(size+1)
    self.name = name
    self.flat_size = size + 1
  
  def __call__(self, x):
    if 0 > x or x >= self.space.n:
      warning("%d out of bounds in discrete space \"%s\"" % (x, self.name))
      x = self.size
    return x

  def write(self, x, array, offset):
    array[offset + self(x)] = 1
  
  def embed(self, x):
    return tf.one_hot(x, self.flat_size)

  def make_ph(self, batch_shape):
    return tf.placeholder(tf.int64, batch_shape, self.name)

class StructConv(Conv):
  def __init__(self, spec, name="StructConv"):
    self.name = name
    self.spec = [(key, f(name=name + '/' + key)) for key, f in spec]
    self.space = spaces.Dict(OrderedDict(
        (name, conv.space) for name, conv in self.spec))
    self.flat_size = sum(conv.flat_size for _, conv in self.spec)
  
  def __call__(self, struct):
    return {name: conv(getattr(struct, name)) for name, conv in self.spec}

  def write(self, struct, array, offset):
    for name, conv in self.spec:
      conv.write(getattr(struct, name), array, offset)
      offset += conv.flat_size

  def embed(self, struct):
    if isinstance(struct, dict):
      return tf.concat([
          conv.embed(struct[name])
          for name, conv in self.spec], -1)
    if hasattr(struct, '_fields'):
      return tf.concat([
          conv.embed(getattr(struct, name))
          for name, conv in self.spec], -1)
    raise TypeError("Unknown struct %s" % struct)

  def make_ph(self, batch_shape):
    return {name: conv.make_ph(batch_shape) for name, conv in self.spec}

class ArrayConv:
  def __init__(self, mk_conv, permutation, name="ArrayConv"):
    self.permutation = [(i, mk_conv(name=name + '/' + str(i))) for i in permutation]
    self.space = spaces.Tuple([conv.space for _, conv in self.permutation])
    self.flat_size = sum(conv.flat_size for _, conv in self.permutation)
  
  def __call__(self, array):
    return [conv(array[i]) for i, conv in self.permutation]

  def write(self, raw_array, array, offset):
    for i, conv in self.permutation:
      conv.write(raw_array[i], array, offset)
      offset += conv.flat_size

  def embed(self, array):
    return tf.concat([
        conv.embed(array[i])
        for i, conv in self.permutation], -1)

  def make_ph(self, batch_shape):
    phs = [None] * len(self.permutation)
    for i, conv in self.permutation:
      phs[i] = conv.make_ph(batch_shape)
    return tuple(phs)

max_char_id = 32 # should be large enough?

max_action_state = 0x017E
num_action_states = 1 + max_action_state

xy_conv = partial(symmetric_conv, 300, 3)
frame_conv = partial(positive_conv, 180)

# generally less than 1 in magnitude
# side-B reaches 18
speed_conv = partial(symmetric_conv, 20)

hitstun_frames_left_conv = partial(SumConv, [
    ('default', frame_conv),
    ('negative', partial(RealConv, (-5, 0), (-1, 0))),
    ('high_falcon', partial(ExceptionConv, [219, 220])),
])

action_frame_conv = partial(SumConv, [
    ('default', frame_conv),
    ('exception', partial(ExceptionConv, [-1, -2])),
])

shield_conv = partial(positive_conv, 60)
damage_conv = partial(RealConv, (0, 1000), (0, 10))

character_conv = partial(DiscreteConv, max_char_id)
action_state_conv = partial(DiscreteConv, num_action_states)

player_spec = [
  ('percent', damage_conv),
  ('facing', partial(symmetric_conv, 1)),
  ('x', xy_conv),
  ('y', xy_conv),
  ('action_state', action_state_conv),
  ('action_frame', action_frame_conv),
  ('character', character_conv),
  ('invulnerable', BoolConv),
  ('hitlag_frames_left', frame_conv),
  ('hitstun_frames_left', hitstun_frames_left_conv),
  ('jumps_used', partial(DiscreteConv, 8)),
  ('charging_smash', BoolConv),
  ('in_air', BoolConv),
  ('speed_air_x_self', speed_conv),
  ('speed_ground_x_self', speed_conv),
  ('speed_y_self', speed_conv),
  ('speed_x_attack', speed_conv),
  ('speed_y_attack', speed_conv),
  ('shield_size', shield_conv),
]

player_conv = partial(StructConv, player_spec)
stage_conv = partial(DiscreteConv, 32)

def make_game_spec(self=0, enemy=1, swap=False):
  players = [self, enemy]
  if swap:
    players.reverse()
  
  return [
    ('players', partial(ArrayConv, player_conv, players)),
    ('stage', stage_conv),
  ]

# maps pid to Conv
game_conv_list = [
  StructConv(make_game_spec(swap=False), name='game-0'),
  StructConv(make_game_spec(swap=True), name='game-1'),
]

CONVS = {}

def get_phillip_conv(pid):
  return game_conv_list[pid]

CONVS['phillip'] = get_phillip_conv

# reduced specs for slippi data
slippi_player_spec = [
  ('x', xy_conv),
  ('y', xy_conv),
  ('character', character_conv),
  ('action_state', action_state_conv),
  # ('action_frame', action_frame_conv),
  ('damage', damage_conv),
  ('shield', shield_conv),
]

slippi_player_conv = partial(StructConv, slippi_player_spec)

slippi_game_spec = [
  ('players', partial(ArrayConv, slippi_player_conv, [0, 1])),
  ('stage', stage_conv),
]

slippi_game_conv = StructConv(slippi_game_spec, name='slippi')

def get_slippi_conv(pid):
  assert pid == 0
  return slippi_game_conv

CONVS['slippi'] = get_slippi_conv

