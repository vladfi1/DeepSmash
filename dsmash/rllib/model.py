import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt

from ray import rllib
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models import lstm

from dsmash import ssbm_actions
from dsmash.rllib import ssbm_spaces

class DelayedActionModel(rllib.models.Model):

  def _build_layers_v2(self, input_dict, num_outputs, options):
    delay = options["custom_options"]["delay"]
    assert(delay > 0)
    
    self.state_init = np.zeros([delay-1], np.int64)

    if not self.state_in:
      self.state_in = tf.placeholder(tf.int64, [None, delay-1], name="delayed_actions")

    delayed_actions = tf.concat([
        self.state_in,
        tf.expand_dims(input_dict["prev_actions"], 1)
    ], axis=1)

    self.state_out = delayed_actions[:, 1:]
    
    embedded_delayed_actions = tf.one_hot(delayed_actions, num_outputs)
    embedded_delayed_actions = snt.MergeDims(1, 2)(embedded_delayed_actions)

    trunk = snt.nets.MLP(
        output_sizes=options["fcnet_hiddens"],
        activation=getattr(tf.nn, options["fcnet_activation"]),
        activate_final=True)

    inputs = tf.concat([input_dict["obs"], embedded_delayed_actions], 1)
    trunk_outputs = trunk(input_dict["obs"])
    
    logits = snt.Linear(num_outputs)(trunk_outputs)
    
    return logits, trunk_outputs


class HumanActionModel(rllib.models.Model, snt.AbstractModule):

  def __init__(self, *args, name="HumanActionModel", imitation=False, **kwargs):
    snt.AbstractModule.__init__(self, name=name)
    self._imitation = imitation
    with self._enter_variable_scope():
      self._value_head = snt.Linear(1, name="value_head")

    rllib.models.Model.__init__(self, *args, **kwargs)

  def _build_layers_v2(self, input_dict, num_outputs, options):
    # make things time major?
    return self(input_dict, num_outputs, options)

  def _build(self, input_dict, num_outputs, options):
    if options.get("use_lstm"):
      cell_size = options.get("lstm_cell_size")
      self.state_init = (
          np.zeros([cell_size], np.float32),
      )
      self.state_in = (
          tf.placeholder(tf.float32, [None, cell_size], name="state_in"),
      )
    else:
      self.state_init = ()
      self.state_in = ()

    if self._imitation:
      obs_embed = ssbm_spaces.slippi_conv_list[0].embed(input_dict["obs"])
      prev_actions = input_dict["prev_actions"]
    else:
      obs_embed = input_dict["obs"]
      prev_actions = tf.unstack(input_dict["prev_actions"], axis=-1)
    
    action_config = nest.flatten(ssbm_actions.repeated_simple_controller_config)
    prev_actions_embed = tf.concat([
        conv.embed(action) for conv, action
        in zip(action_config, prev_actions)], -1)
    
    prev_rewards_embed = tf.expand_dims(input_dict["prev_rewards"], -1)
    inputs = tf.concat([obs_embed, prev_actions_embed, prev_rewards_embed], -1)

    trunk = snt.nets.MLP(
        output_sizes=options["fcnet_hiddens"],
        activation=getattr(tf.nn, options["fcnet_activation"]),
        activate_final=True)

    if not self._imitation:
      inputs = lstm.add_time_dimension(inputs, self.seq_lens)

    trunk_outputs = snt.BatchApply(trunk)(inputs)
    
    if options.get("use_lstm"):
      gru = snt.GRU(cell_size)
      core_outputs, state_out = tf.nn.dynamic_rnn(
          gru,
          trunk_outputs,
          initial_state=self.state_in[0],
          sequence_length=None if self._imitation else self.seq_lens,
          time_major=self._imitation)
      self.state_out = [state_out]
    else:
      core_outputs = trunk_outputs
      self.state_out = []

    self._logit_head = snt.Linear(num_outputs, name="logit_head")
    logits = snt.BatchApply(self._logit_head)(core_outputs)
    self.values = tf.squeeze(snt.BatchApply(self._value_head)(core_outputs), -1)
    
    return logits, core_outputs

  def value_function(self):
    return snt.MergeDims(0, 2)(self.values)


def register():
  ModelCatalog.register_custom_model("delayed_action", DelayedActionModel)
  ModelCatalog.register_custom_model("human_action", HumanActionModel)

