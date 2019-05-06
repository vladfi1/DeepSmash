import time
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import sonnet as snt
import trfl

import gym
from gym import spaces

from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule

from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.agents import impala, Trainer, trainer
from ray.rllib.agents.impala import vtrace
from ray.rllib.agents.impala.vtrace_policy_graph import VTracePolicyGraph

from dsmash import util
from dsmash import ssbm_actions
from dsmash.rllib import ssbm_spaces, imitation_env
from dsmash.rllib.model import HumanActionModel


class HumanPolicyGraph(VTracePolicyGraph):
  """PolicyGraph compatible with imitation learning.
  
  Only works with HumanActionTrainer and HumanActionModel.
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
    config = dict(DEFAULT_CONFIG, **config)
    assert config["batch_mode"] == "truncate_episodes", \
      "Must use `truncate_episodes` batch mode with V-trace."
    self.config = config

    self.sess = tf.get_default_session()
    self.grads = None
    
    imitation = config["imitation"]
    assert not imitation
    
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

    assert is_multidiscrete

    if imitation:
      make_action_ph = lambda: ssbm_actions.make_ph(
          ssbm_actions.flat_repeated_config, batch_shape)
      actions = make_action_ph()
      prev_actions = make_action_ph()
    else:  # actions are stacked "multidiscrete"
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
      behavior_logp = tf.placeholder(tf.float32, batch_shape)

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

    # HumanActionModel doesn't flatten outputs
    flat_outputs = snt.MergeDims(0, 2)(self.model.outputs)

    if autoregressive:
      action_dist = ssbm_actions.AutoRegressive(
          nest.map_structure(
              lambda conv: conv.build_dist(),
              ssbm_actions.flat_repeated_config),
          residual=config.get("residual"))
      actions_logp, actions_entropy = action_dist.logp(
          flat_outputs, tf.unstack(actions, axis=-1))
      action_sampler, self.sampled_logp = action_dist.sample(flat_outputs)
      action_sampler = tf.stack([
          tf.cast(t, tf.int64) for t in nest.flatten(action_sampler)], axis=-1)
      sampled_prob = tf.exp(self.sampled_logp)
    else:
      dist_inputs = tf.split(flat_outputs, output_hidden_shape, axis=-1)
      action_dist = dist_class(dist_inputs)
      int64_actions = [tf.cast(x, tf.int64) for x in actions]
      actions_logp = action_dist.logp(int64_actions)
      actions_entropy = action_dist.entropy()
      action_sampler = action_dist.sample()
      sampled_prob = action_dist.sampled_action_prob()
      self.sampled_logp = tf.log(sampled_prob)

    self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                      tf.get_variable_scope().name)

    def make_time_major(tensor, drop_last=False):
      """Swaps batch and trajectory axis.
      Args:
        tensor: A tensor or list of tensors to reshape.
        drop_last: A bool indicating whether to drop the last
        trajectory item.
      Returns:
        res: A tensor with swapped axes or a list of tensors with
        swapped axes.
      """
      if isinstance(tensor, list):
        return [make_time_major(t, drop_last) for t in tensor]

      if self.model.state_init:
        B = tf.shape(self.model.seq_lens)[0]
        T = tf.shape(tensor)[0] // B
      else:
        # Important: chop the tensor into batches at known episode cut
        # boundaries. TODO(ekl) this is kind of a hack
        T = self.config["sample_batch_size"]
        B = tf.shape(tensor)[0] // T
      rs = tf.reshape(tensor,
              tf.concat([[B, T], tf.shape(tensor)[1:]], axis=0))

      # swap B and T axes
      res = tf.transpose(
        rs,
        [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))))

      if drop_last:
        return res[:-1]
      return res

    # actual loss computation
    values_tm = make_time_major(self.model.value_function())
    baseline_values = values_tm[:-1]
    actions_logp_tm = make_time_major(actions_logp)
    behavior_logp_tm = make_time_major(behavior_logp)
    log_rhos_tm = actions_logp_tm - behavior_logp_tm

    discounts = tf.fill(tf.shape(baseline_values), config["gamma"])
    if not config.get("soft_horizon"):
      discounts *= tf.to_float(~make_time_major(dones, True))
    
    vtrace_returns = vtrace.from_importance_weights(
        log_rhos=log_rhos_tm[:-1],
        discounts=discounts,
        rewards=make_time_major(rewards, True),
        values=baseline_values,
        bootstrap_value=values_tm[-1])

    vf_loss = tf.reduce_mean(tf.squared_difference(
        vtrace_returns.vs, baseline_values))
    pi_loss = tf.reduce_mean(actions_logp_tm * vtrace_returns.pg_advantages)
    entropy_mean = tf.reduce_mean(actions_entropy)

    total_loss = pi_loss
    total_loss += self.config["vf_loss_coeff"] * vf_loss
    total_loss -= self.config["entropy_coeff"] * entropy_mean
    self.total_loss = total_loss        

    kl_mean = -tf.reduce_mean(log_rhos_tm)

    # Initialize TFPolicyGraph
    loss_in = [
      (SampleBatch.ACTIONS, actions),
      (SampleBatch.DONES, dones),
      ("behavior_logp", behavior_logp),
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
        "pi_loss": pi_loss,
        "entropy": entropy_mean,
        "grad_gnorm": tf.global_norm(self._grads),
        "var_gnorm": tf.global_norm(self.var_list),
        "vf_loss": vf_loss,
        "vf_explained_var": explained_variance(
          tf.reshape(vtrace_returns.vs, [-1]),
          tf.reshape(baseline_values, [-1])),
        "kl_mean": kl_mean,
      },
    }

  @override(TFPolicyGraph)
  def copy(self, existing_inputs):
    raise NotImplementedError

  @override(TFPolicyGraph)
  def extra_compute_action_fetches(self):
    return dict(
        TFPolicyGraph.extra_compute_action_fetches(self),
        **{"behavior_logp": self.sampled_logp})

  @override(PolicyGraph)
  def postprocess_trajectory(self,
                             sample_batch,
                             other_agent_batches=None,
                             episode=None):
    # not used, so save some bandwidth
    del sample_batch.data[SampleBatch.NEXT_OBS]
    return sample_batch

DEFAULT_CONFIG = trainer.with_base_config(impala.DEFAULT_CONFIG, {
  "data_path": None,
  "autoregressive": False,
  "residual": False,
  "imitation": False,
})


class HumanTrainer(impala.ImpalaTrainer):
  """VTrace with human action space.."""
  
  _name = "HumanAction"
  _default_config = DEFAULT_CONFIG
  _policy_graph = HumanPolicyGraph
  
  def __setstate__(self, state):
    self.local_evaluator.policy_map[DEFAULT_POLICY_ID].set_state(state)

  def __getstate__(self):
    self.local_evaluator.policy_map[DEFAULT_POLICY_ID].get_state()

