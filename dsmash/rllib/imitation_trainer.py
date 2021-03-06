import time
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt
import trfl

import gym
from gym import spaces

from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.evaluation.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule

from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.agents import impala, Trainer, trainer
from ray.rllib.agents.impala.vtrace_policy_graph import VTracePolicyGraph

from dsmash import util
from dsmash import ssbm_actions
from dsmash.rllib import ssbm_spaces, imitation_env
from dsmash.rllib.model import HumanActionModel

from tensorflow.python.client import device_lib

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(get_available_gpus())


class ImitationPolicyGraph(VTracePolicyGraph):
  """PolicyGraph specialized for imitation learning.
  
  Only works with ImitationTrainer and HumanActionModel.
  ImitationTrainer creates pseudo-SampleBatches which are time-major nests.
  
  TODO: preserve compatibility with ImitationEnv
  """

  def __init__(self,
         observation_space,
         action_space,
         config,
         existing_inputs=None):
    #with tf.device("/gpu:1"):
    self._init_helper(observation_space, action_space, config, existing_inputs)

  def _init_helper(self,
         observation_space,
         action_space,
         config,
         existing_inputs=None):
    print(get_available_gpus())
    config = dict(impala.impala.DEFAULT_CONFIG, **config)
    assert config["batch_mode"] == "truncate_episodes", \
      "Must use `truncate_episodes` batch mode with V-trace."
    self.config = config

    self.sess = tf.get_default_session()
    self.grads = None
    
    imitation = config["imitation"]
    
    if imitation:
      T = config["sample_batch_size"]
      B = config["train_batch_size"] // T
      batch_shape = (T, B)
    else:
      batch_shape = (None,)

    if isinstance(action_space, gym.spaces.Discrete):
      is_multidiscrete = False
      actions_shape = batch_shape
      output_hidden_shape = [action_space.n]
    elif isinstance(action_space, gym.spaces.multi_discrete.MultiDiscrete):
      is_multidiscrete = True
      actions_shape = batch_shape + (len(action_space.nvec),)
      output_hidden_shape = action_space.nvec.astype(np.int32)
    else:
      raise UnsupportedSpaceException(
        "Action space {} is not supported for IMPALA.".format(
          action_space))

    if imitation:
      make_action_ph = lambda: ssbm_actions.make_ph(
          ssbm_actions.flat_repeated_config, batch_shape)
      actions = make_action_ph()
      prev_actions = make_action_ph()
    else:
      actions = tf.placeholder(tf.int64, actions_shape, name="actions")
      prev_actions = tf.placeholder(tf.int64, actions_shape, name="prev_actions")

    # Create input placeholders
    dones = tf.placeholder(tf.bool, batch_shape, name="dones")
    rewards = tf.placeholder(tf.float32, batch_shape, name="rewards")
    if imitation:
      observations = ssbm_spaces.slippi_conv_list[0].make_ph(batch_shape)
    else:
      observations = tf.placeholder(
          tf.float32, [None] + list(observation_space.shape))

    existing_state_in = None
    existing_seq_lens = None

    # Setup the policy
    autoregressive = config.get("autoregressive")
    if autoregressive:
      logit_dim = 128  # not really logits
    else:
      dist_class, logit_dim = ModelCatalog.get_action_dist(
        action_space, self.config["model"])

    prev_rewards = tf.placeholder(tf.float32, batch_shape, name="prev_reward")
    self.model = HumanActionModel(
      {
        "obs": observations,
        "prev_actions": prev_actions,
        "prev_rewards": prev_rewards,
        "is_training": self._get_is_training_placeholder(),
      },
      observation_space,
      action_space,
      logit_dim,
      self.config["model"],
      imitation=imitation,
      state_in=existing_state_in,
      seq_lens=existing_seq_lens)

    if autoregressive:
      action_dist = ssbm_actions.AutoRegressive(
          nest.map_structure(
              lambda conv: conv.build_dist(),
              ssbm_actions.flat_repeated_config),
          residual=config.get("residual"))
      actions_logp = snt.BatchApply(action_dist.logp)(
          self.model.outputs, actions)
      action_sampler, sampled_logp = snt.BatchApply(
          action_dist.sample)(self.model.outputs)
      sampled_prob = tf.exp(sampled_logp)
    else:
      dist_inputs = tf.split(self.model.outputs, output_hidden_shape, axis=-1)
      action_dist = dist_class(snt.MergeDims(0, 2)(dist_inputs))
      int64_actions = [tf.cast(x, tf.int64) for x in actions]
      actions_logp = action_dist.logp(snt.MergeDims(0, 2)(int64_actions))
      action_sampler = action_dist.sample()
      sampled_prob = action_dist.sampled_action_prob(),

    self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                      tf.get_variable_scope().name)

    # actual loss computation
    imitation_loss = -tf.reduce_mean(actions_logp)
    
    tm_values = self.model.values
    baseline_values = tm_values[:-1]
    
    if config.get("soft_horizon"):
      discounts = config["gamma"]
    else:
      discounts = tf.to_float(~dones[:-1]) * config["gamma"]
    
    td_lambda = trfl.td_lambda(
        state_values=baseline_values,
        rewards=rewards[:-1],
        pcontinues=discounts,
        bootstrap_value=tm_values[-1],
        lambda_=config.get("lambda", 1.))

    # td_lambda.loss has shape [B] after a reduce_sum
    vf_loss = tf.reduce_mean(td_lambda.loss) / T
    
    self.total_loss = imitation_loss + self.config["vf_loss_coeff"] * vf_loss

    # Initialize TFPolicyGraph
    loss_in = [
      (SampleBatch.ACTIONS, actions),
      (SampleBatch.DONES, dones),
      # (BEHAVIOUR_LOGITS, behaviour_logits),
      (SampleBatch.REWARDS, rewards),
      (SampleBatch.CUR_OBS, observations),
      (SampleBatch.PREV_ACTIONS, prev_actions),
      (SampleBatch.PREV_REWARDS, prev_rewards),
    ]
    LearningRateSchedule.__init__(self, self.config["lr"],
                    self.config["lr_schedule"])
    TFPolicyGraph.__init__(
      self,
      observation_space,
      action_space,
      self.sess,
      obs_input=observations,
      action_sampler=action_sampler,
      action_prob=sampled_prob,
      loss=self.total_loss,
      model=self.model,
      loss_inputs=loss_in,
      state_inputs=self.model.state_in,
      state_outputs=self.model.state_out,
      prev_action_input=prev_actions,
      prev_reward_input=prev_rewards,
      seq_lens=self.model.seq_lens,
      max_seq_len=self.config["model"]["max_seq_len"],
      batch_divisibility_req=self.config["sample_batch_size"])

    self._loss_input_dict = dict(
        self._loss_inputs, state_in=self._state_inputs)

    self.sess.run(tf.global_variables_initializer())

    self.stats_fetches = {
      LEARNER_STATS_KEY: {
        "cur_lr": tf.cast(self.cur_lr, tf.float64),
        "imitation_loss": imitation_loss,
        #"entropy": self.loss.entropy,
        "grad_gnorm": tf.global_norm(self._grads),
        "var_gnorm": tf.global_norm(self.var_list),
        "vf_loss": vf_loss,
        "vf_explained_var": explained_variance(
          tf.reshape(td_lambda.extra.discounted_returns, [-1]),
          tf.reshape(baseline_values, [-1])),
      },
      "state_out": self.model.state_out,
    }

  @override(VTracePolicyGraph)
  def _get_loss_inputs_dict(self, batch):
    feed = {}
    def add_feed(ph, val):
      feed[ph] = val
    util.deepZipWith(add_feed, self._loss_input_dict, batch)
    return feed

DEFAULT_CONFIG = trainer.with_base_config(impala.DEFAULT_CONFIG, {
  "data_path": None,
  "autoregressive": False,
  "residual": False,
  "imitation": True,
})

class DummyTrainer(impala.ImpalaTrainer):
  """Imitation learning."""
  
  _name = "IMITATION"
  _default_config = DEFAULT_CONFIG
  _policy_graph = ImitationPolicyGraph
  
  def __setstate__(self, state):
    self.local_evaluator.policy_map[DEFAULT_POLICY_ID].set_state(state)

class ImitationTrainer(impala.ImpalaTrainer):
  """Imitation learning."""
  
  _name = "IMITATION"
  _default_config = DEFAULT_CONFIG
  _policy_graph = ImitationPolicyGraph
  

  @override(impala.ImpalaTrainer)
  def _init(self, config, env_creator):
    self.sess = tf.Session(config=tf.ConfigProto(**config["tf_session_args"]))

    with self.sess.as_default():
      self.policy_graph = ImitationPolicyGraph(
        ssbm_spaces.slippi_conv_list[0].space,
        ssbm_actions.repeated_simple_controller_space,
        config)

    train_batches = config["train_batch_size"] // config["sample_batch_size"]
    tile = lambda x: np.array([x] * train_batches)
    self.state_init = nest.map_structure(
        tile, self.policy_graph.model.state_init)
    self._readers = [
        imitation_env.GameReader(config["data_path"])
        for _ in range(train_batches)]
  
  def __getstate__(self):
    return self.policy_graph.get_state()
  
  def __setstate__(self, state):
    self.policy_graph.set_state(state)

  def train_step(self):
    sample_batches = [
        reader.get_sample_batch(self.config["sample_batch_size"])
        for reader in self._readers]
    sample_batch = nest.map_structure(
        lambda *xs: np.stack(xs, 1), *sample_batches)
    sample_batch["state_in"] = self.state_init
    stats_fetches = self.policy_graph.learn_on_batch(sample_batch)
    self.state_init = stats_fetches["state_out"]
    return stats_fetches

  @override(Trainer)
  def _train(self):
    start = time.time()
    stats_fetches = self.train_step()
    del stats_fetches["state_out"]
    steps = 1
    while time.time() - start < self.config["min_iter_time_s"]:
      self.train_step()
      steps += 1
    throughput = self.config["train_batch_size"] * steps / (time.time() - start)
    result = {}
    result.update(
        timesteps_this_iter=steps,
        train_throughput=throughput,
        **stats_fetches)
    return result

