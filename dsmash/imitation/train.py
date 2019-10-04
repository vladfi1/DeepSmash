import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt

from dsmash import ssbm_actions, ssbm_spaces
from dsmash import util
from dsmash.default import Default, Option
from dsmash.imitation import read_data

DEFAULT_MODEL_OPTIONS = dict(
    fcnet_hiddens=[256],
    fcnet_activation="relu",
    core_size=256,
)


class HumanActionModel(snt.AbstractModule):
  
  def __init__(self, options):
    super(HumanActionModel, self).__init__(name="Model")
    self._options = options
    self._conv = ssbm_spaces.slippi_conv_list[0]
    
    with self._enter_variable_scope():
      self._trunk = snt.nets.MLP(
          output_sizes=options["fcnet_hiddens"],
          activation=getattr(tf.nn, options["fcnet_activation"]),
          activate_final=True)
      
      self._core = snt.LSTM(options["core_size"])
      self.initial_state = self._core.initial_state
      
      self._action_dist = ssbm_actions.AutoRegressive(
          nest.map_structure(
              lambda conv: conv.build_dist(),
              ssbm_actions.repeated_simple_controller_config),
          residual=True)


  def get_phs(self, unroll_length, batch_size):
    batch_shape = (unroll_length, batch_size)
    make_action_ph = lambda: util.nt_to_dict(ssbm_actions.make_ph(
        ssbm_actions.repeated_simple_controller_config, batch_shape))

    input_dict = dict(
        obs = self._conv.make_ph(batch_shape),
        actions = make_action_ph(),
        prev_actions = make_action_ph(),
        dones = tf.placeholder(tf.bool, batch_shape, "dones"),
        rewards = tf.placeholder(tf.float32, batch_shape, "prev_rewards"),
        prev_rewards = tf.placeholder(tf.float32, batch_shape, "prev_rewards"),
    )

    core_states = nest.map_structure(
        lambda ts: tf.placeholder(tf.float32, [batch_size] + ts.as_list()),
        self._core.state_size)

    return input_dict, core_states

  def _build(self, input_dict, prev_core_states):
    obs_embed = self._conv.embed(input_dict["obs"])
    prev_actions_embed = self._action_dist.embed(input_dict["prev_actions"])
    prev_rewards_embed = tf.expand_dims(input_dict["prev_rewards"], -1)
    inputs = tf.concat([obs_embed, prev_actions_embed, prev_rewards_embed], -1)

    trunk_outputs = snt.BatchApply(self._trunk)(inputs)
    
    core_outputs, next_core_states = tf.nn.dynamic_rnn(
        self._core,
        trunk_outputs,
        initial_state=prev_core_states,
        time_major=True)

    return core_outputs, next_core_states

  def train(self, input_dict, prev_core_states):
    core_outputs, next_core_states = self(input_dict, prev_core_states)
    actions_logp, _ = snt.BatchApply(self._action_dist.logp)(
        core_outputs, input_dict["actions"])
    
    to_log = {}
    mean_action_logp = tf.reduce_mean(actions_logp)
    to_log["mean_action_logp"] = mean_action_logp
    loss = -mean_action_logp

    return loss, next_core_states, to_log


class Trainer(Default):
  _options = [
      Option("data_path", nargs="+", type=str, help="path to pickled games"),
      Option("unroll_length", type=int, default=40),
      Option("batch_size", type=int, default=256),
      Option("learning_rate", type=float, default=2e-4),
      Option("log_interval", type=int, default=30,
             help="log interval in seconds"),
  ]

  def __init__(self, **kwargs):
    Default.__init__(self, **kwargs)
    self._sess = tf.Session()

    self._model = HumanActionModel(DEFAULT_MODEL_OPTIONS)
    self._core_states = self._sess.run(
        self._model.initial_state(self.batch_size))

    self._readers = [
        read_data.GameReader(self.data_path)
        for _ in range(self.batch_size)]

    self._placeholders = self._model.get_phs(
        self.unroll_length, self.batch_size)   

    loss, next_core_state, to_log = self._model.train(*self._placeholders)
    
    self._optimizer = tf.train.AdamOptimizer(self.learning_rate)
    train_op = self._optimizer.minimize(loss)
    self._fetches = train_op, next_core_state, to_log

    self._sess.run(tf.global_variables_initializer())

  def train_step(self):
    sample_batches = [
        reader.get_sample_batch(self.unroll_length)
        for reader in self._readers]
    input_dict = nest.map_structure(
        lambda *xs: np.stack(xs, 1), *sample_batches)

    # self._model._action_dist.validate(input_dict["actions"])

    feed_dict = {}
    # nest.map_structure doesn't work here because the input_dict may have
    # extra keys (like action_frame) that we don't use as inputs
    util.deepZipWith(  
        lambda ph, x: feed_dict.__setitem__(ph, x),
        self._placeholders, (input_dict, self._core_states))

    _, next_core_states, to_log = self._sess.run(self._fetches, feed_dict)
    self._core_states = next_core_states
    return to_log

  def train(self):
    start_time = time.time()
    last_logged = start_time
    iters = 0
    steps_per_iter = self.unroll_length * self.batch_size
    while True:
      to_log = self.train_step()
      iters += 1

      current_time = time.time()
      if current_time - last_logged > self.log_interval:
        total_time = current_time - start_time
        total_steps = steps_per_iter * iters
        print(to_log)
        print("SPS: ", int(total_steps / total_time))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  Trainer.update_parser(parser)
  args = parser.parse_args()
  trainer = Trainer(**args.__dict__)
  trainer.train()

