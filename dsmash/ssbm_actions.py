from collections import namedtuple
import enum
from functools import partial

import numpy as np
from gym import spaces
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt

from dsmash.slippi.types import *


class Convertor:
  def to_raw(self, x):
    return x

class DiscreteFloat(Convertor):
  """Discretizes continuous scalars."""
  def __init__(self, values):
    self._values = np.array(values, dtype=np.float32)
    self._cutoffs = (self._values[:-1] + self._values[1:]) / 2
  
  @property
  def size(self):
    return len(self._values)
  
  def to_discrete(self, x):
    return np.searchsorted(self._cutoffs, x)

  def to_raw(self, i):
    return self._values[i]

  def embed(self, i):
    return tf.expand_dims(tf.gather(self._values, i), -1)
  
  def build_dist(self):
    return Categorical(self.size)
  
  def space(self):
    return spaces.Discrete(self.size)
  
  def make_ph(self, batch_shape):
    return tf.placeholder(tf.int64, batch_shape)

class Discrete(Convertor):
  
  def __init__(self, size):
    self.size = size

  def to_raw(self, i):
    assert 0 <= i and i < self.size
    return i

  def embed(self, i):
    return tf.one_hot(i, self.size)

  def build_dist(self):
    return Categorical(self.size)

  def space(self):
    return spaces.Discrete(self.size)

  def make_ph(self, batch_shape):
    return tf.placeholder(tf.int64, batch_shape)

class Binary(Convertor):

  def embed(self, b):
    return tf.expand_dims(tf.to_float(b), -1)

  def to_raw(self, i):
    assert i in [0, 1]
    return bool(i)

  def build_dist(self):
    return Bernoulli()

  def space(self):
    return spaces.Discrete(2)

  def make_ph(self, batch_shape):
    return tf.placeholder(tf.bool, batch_shape)

BINARY = Binary()

discrete_trigger = DiscreteFloat([0, 0.5, 1])
discrete_stick = DiscreteFloat([-1, -0.5, -0.2, 0, 0.2, 0.5, 1])

buttons_config = SimpleButtons(**{b: BINARY for b in simple_buttons})

stick_config = Stick(x=discrete_stick, y=discrete_stick)

simple_controller_config = SimpleController(
  buttons=buttons_config,
  joystick=stick_config,
  cstick=Discrete(len(SimpleCStick)),
  trigger=discrete_trigger)

repeated_simple_controller_config = RepeatedAction(
  action = simple_controller_config,
  repeat = Discrete(15)
)

flat_repeated_config = nest.flatten(repeated_simple_controller_config)

def to_raw(config, value):
  return nest.map_structure(lambda conv, x: conv.to_raw(x), config, value)

def to_multidiscrete(config):
  return spaces.MultiDiscrete([c.space().n for c in nest.flatten(config)])

repeated_simple_controller_space = to_multidiscrete(
    repeated_simple_controller_config)

def make_ph(config, batch_shape):
  return nest.map_structure(lambda conv: conv.make_ph(batch_shape), config)


class Dist(snt.AbstractModule):

  def _build(self, inputs):
    return self.sample(inputs)


class Bernoulli(Dist):
  def __init__(self, name='Bernoulli'):
    super(Bernoulli, self).__init__(name=name)
    
    with self._enter_variable_scope():
      self._linear = snt.Linear(1)

  def sample(self, inputs):
    logits = tf.squeeze(self._linear(inputs), -1)
    probs = tf.nn.sigmoid(logits)
    uniform = tf.random.uniform(tf.shape(probs))
    sample = tf.less(uniform, probs)
    logp = -tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(sample),
        logits=logits)
    return sample, logp
  
  def logp(self, inputs, sample):
    logits = tf.squeeze(self._linear(inputs), -1)
    return -tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(sample),
        logits=logits)  

  def embed(self, sample):
    return tf.expand_dims(tf.to_float(sample), -1)


class Categorical(Dist):
  def __init__(self, size, name='Categorical'):
    super(Categorical, self).__init__(name=name)
    self._size = size
    
    with self._enter_variable_scope():
      self._linear = snt.Linear(size)
    
  def sample(self, inputs):
    logits = self._linear(inputs)
    sample = tf.squeeze(tf.random.categorical(logits, 1), -1)
    logp = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=sample)
    return sample, logp
  
  def logp(self, inputs, x):
    logits = self._linear(inputs)
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=x)
  
  def embed(self, x):
    return tf.one_hot(x, self._size)

class AutoRegressive(Dist):

  def __init__(self, dist_struct, name='AutoRegressive'):
    super(AutoRegressive, self).__init__(name=name)
    self._dist_struct = dist_struct
    self._dist_flat = nest.flatten(dist_struct)

  def sample(self, inputs):
    samples = []
    logps = []
    
    for dist in self._dist_flat:
      sample, logp = dist.sample(inputs)
      samples.append(sample)
      logps.append(logp)
      sample_repr = dist.embed(sample)
      inputs = tf.concat([inputs, sample_repr], -1)

    sample_struct = nest.pack_sequence_as(self._dist_struct, samples)
    logp = tf.add_n(logps)
    return sample_struct, logp

  def logp(self, inputs, sample_struct):
    logps = []
    
    for sample, dist in zip(nest.flatten(sample_struct), self._dist_flat):
      logp = dist.logp(inputs, sample)
      logps.append(logp)
      sample_repr = dist.embed(sample)
      inputs = tf.concat([inputs, sample_repr], -1)
    
    return tf.add_n(logps)

  def embed(self, sample_struct):
    embeddings = []
    for sample, dist in zip(nest.flatten(sample_struct), self._dist_flat):
      embeeddings.append(dist.embed(sample))
    return tf.concat(embeddings, -1)

