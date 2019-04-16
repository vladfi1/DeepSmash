import numpy as np
import tensorflow as tf
import trfl

import gym

from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule

from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.agents import impala
from ray.rllib.agents.impala.vtrace_policy_graph import VTracePolicyGraph


class ImitationPolicyGraph(VTracePolicyGraph):
  def __init__(self,
         observation_space,
         action_space,
         config,
         existing_inputs=None):
    config = dict(impala.impala.DEFAULT_CONFIG, **config)
    assert config["batch_mode"] == "truncate_episodes", \
      "Must use `truncate_episodes` batch mode with V-trace."
    self.config = config
    self.sess = tf.get_default_session()
    self.grads = None

    if isinstance(action_space, gym.spaces.Discrete):
      is_multidiscrete = False
      actions_shape = [None]
      output_hidden_shape = [action_space.n]
    elif isinstance(action_space, gym.spaces.multi_discrete.MultiDiscrete):
      is_multidiscrete = True
      actions_shape = [None, len(action_space.nvec)]
      output_hidden_shape = action_space.nvec.astype(np.int32)
    else:
      raise UnsupportedSpaceException(
        "Action space {} is not supported for IMPALA.".format(
          action_space))

    # Create input placeholders
    if existing_inputs:
      print("EXISTING INPUTS")
      actions, dones, behaviour_logits, rewards, observations, \
        prev_actions, prev_rewards = existing_inputs[:7]
      existing_state_in = existing_inputs[7:-1]
      existing_seq_lens = existing_inputs[-1]
    else:
      print("NO EXISTING INPUTS")
      actions = tf.placeholder(tf.int64, actions_shape, name="ac")
      dones = tf.placeholder(tf.bool, [None], name="dones")
      rewards = tf.placeholder(tf.float32, [None], name="rewards")
      behaviour_logits = tf.placeholder(
        tf.float32, [None, sum(output_hidden_shape)],
        name="behaviour_logits")
      observations = tf.placeholder(
        tf.float32, [None] + list(observation_space.shape))
      existing_state_in = None
      existing_seq_lens = None

    # Unpack behaviour logits
    unpacked_behaviour_logits = tf.split(
      behaviour_logits, output_hidden_shape, axis=1)

    # Setup the policy
    dist_class, logit_dim = ModelCatalog.get_action_dist(
      action_space, self.config["model"])
    prev_actions = ModelCatalog.get_action_placeholder(action_space)
    prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")
    self.model = ModelCatalog.get_model(
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
      state_in=existing_state_in,
      seq_lens=existing_seq_lens)
    unpacked_outputs = tf.split(
      self.model.outputs, output_hidden_shape, axis=1)

    dist_inputs = unpacked_outputs if is_multidiscrete else \
      self.model.outputs
    action_dist = dist_class(dist_inputs)

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

    if self.model.state_in:
      max_seq_len = tf.reduce_max(self.model.seq_lens) - 1
      mask = tf.sequence_mask(self.model.seq_lens, max_seq_len)
      mask = tf.reshape(mask, [-1])
    else:
      mask = tf.ones_like(rewards, dtype=tf.bool)

    # actual loss computation
    valid_mask = make_time_major(mask, drop_last=True)
    actions_logp = make_time_major(
        action_dist.logp(actions), drop_last=True)
    imitation_loss = -tf.reduce_mean(tf.boolean_mask(actions_logp, valid_mask))
    
    tm_values = make_time_major(self.model.value_function())
    baseline_values = tm_values[:-1]
    tm_dones = make_time_major(dones, drop_last=True)
    td_lambda = trfl.td_lambda(
        state_values=baseline_values,
        rewards=make_time_major(rewards)[:-1],
        pcontinues=tf.to_float(~tm_dones) * config["gamma"],
        bootstrap_value=tm_values[-1],
        lambda_=config.get("lambda", 1.))

    # td_lambda.loss has shape [B] after a reduce_sum
    vf_loss = tf.reduce_mean(td_lambda.loss) / tf.to_float(tf.shape(tm_dones)[0])
    
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
      action_sampler=action_dist.sample(),
      action_prob=action_dist.sampled_action_prob(),
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
    }
  

class ImitationTrainer(impala.ImpalaTrainer):
  """Imitation learning."""
  
  _name = "IMITATION"
  _policy_graph = ImitationPolicyGraph

