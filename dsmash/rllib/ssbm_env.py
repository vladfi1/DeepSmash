from collections import deque
import os
import psutil
import cProfile
import time

import numpy as np
from tensorflow.python.util import nest
import gym
import ray
from ray import rllib

from dsmash import ssbm, ssbm_actions
from dsmash.env.ssbm_env import SSBMEnv
from dsmash.rllib import ssbm_spaces


class FixedActionMap:
  
  def __init__(self, config):
    self._act_every = config.get("act_every", 3)
    action_set = ssbm.actionTypes["custom_sh2_wd"]
    self._action_chains = action_set.get_action_chains(self._act_every)
    
    #self.action_space = None
    self.action_space = gym.spaces.Discrete(action_set.size)    

  def get_controllers(self, act_id, char):
    chain = self._action_chains[act_id]
    return [chain[i].get_real_controller(char) for i in range(self._act_every)]

class SimpleActionMap:
  def __init__(self, config):
    self._action_config = ssbm_actions.simple_controller_config
    self.action_space = ssbm_actions.to_multidiscrete(self._action_config)

  def get_controllers(self, flat_simple_controller, _):
    simple_controller = nest.pack_sequence_as(
        self._action_config, flat_simple_controller.tolist())
    return [ssbm_actions.to_raw(self._action_config, simple_controller)]

class RepeatActionMap:
  def __init__(self, config):
    self._action_config = ssbm_actions.repeated_simple_controller_config
    self.action_space = ssbm_actions.to_multidiscrete(self._action_config)

  def get_controllers(self, flat_repeat_controller, _):
    repeat_controller = nest.pack_sequence_as(
        self._action_config, flat_repeat_controller.tolist())
    return [ssbm_actions.to_raw(self._action_config.action, repeat_controller.action)] * (repeat_controller.repeat + 1)

# TODO: separate action map per agent?
def get_action_map(config):
  mode = config.get('action_mode', 'fixed')
  
  if mode == 'fixed':
    cls = FixedActionMap
  elif mode == 'slippi':
    cls = SimpleActionMap
  elif mode == 'slippi_repeat':
    cls = RepeatActionMap
  else:
    raise TypeError("Unknown mode %s" % mode)
  
  return cls(config)


class ActionQueue:
  
  def __init__(self):
    self._actions = deque()
    self._needs_obs = deque()
  
  def extend(self, actions):
    assert actions
    self._actions.extend(actions)
    self._needs_obs.extend([False] * (len(actions) - 1))
    self._needs_obs.append(True)
  
  def next(self):
    return self._actions.popleft(), self._needs_obs.popleft()


class MultiSSBMEnv(rllib.env.MultiAgentEnv):

  def __init__(self, config):
    print("MultiSSBMEnv", config.keys())
    self._ssbm_config = config["ssbm_config"]
    self._flat_obs = config.get("flat_obs", False)
    self._conv_fn = ssbm_spaces.CONVS[config.get("conv", "phillip")]
    self._steps_this_episode = 0
    self._cpu = config.get('cpu')
    self._profile = config.get('profile', False)
    self._env = None
    self._action_map = get_action_map(config)
    self.action_space = self._action_map.action_space
    print(self.action_space)

    default_conv = self._conv_fn(0)
    if self._flat_obs:
      self.observation_space = gym.spaces.Box(
        low=-10, high=10,
        shape=(default_conv.flat_size,),
        dtype=np.float32)
    else:
      self.observation_space = default_conv.space
    print(self.observation_space)

  def _get_obs(self, pids=None):
    if pids is None:
      pids = self._env.ai_pids
    game_state = self._env.get_state()
    return {pid: self._convs[pid](game_state) for pid in pids}

  def reset(self):
    if self._env is None:
      self._env = SSBMEnv(**self._ssbm_config)
      if self._cpu is not None:
        psutil.Process().cpu_affinity([self._cpu])
        # set cpu affinity on dolphin process and all threads
        os.system('taskset -a -c -p %d %d' % (self._cpu, self._env.dolphin_process.pid))

      if self._profile:
        self._profile_counter = 0
        self._profiler = cProfile.Profile()
        #self._profiler.enable()

      self._action_queues = {pid: ActionQueue() for pid in self._env.ai_pids}
      self._rewards = {pid: 0 for pid in self._env.ai_pids}

      self._convs = {
          pid: self._conv_fn(pid)
          for pid in self._env.ai_pids
      }
      if self._flat_obs:
        self._convs = {pid: conv.make_flat for pid, conv in self._convs.items()}

    return self._get_obs()

  def step(self, actions):
    if self._profile:
      self._profile_counter += 1
      if self._profile_counter % 10000 == 0:
        self._profiler.dump_stats('/tmp/ssbm_stats')
      return self._profiler.runcall(self._step, actions)
    return self._step(actions)

  def _step(self, actions):
    for pid, act_id in actions.items():
      self._action_queues[pid].extend(
          self._action_map.get_controllers(act_id, self._env.characters[pid]))

    obs_pids = []
    
    while not obs_pids:
      multi_action = {}
      for pid, q in self._action_queues.items():
        controller, needs_obs = q.next()
        multi_action[pid] = controller
        if needs_obs:
          obs_pids.append(pid)

      _, step_rewards = self._env.step(multi_action)

      for pid, r in step_rewards.items():
        self._rewards[pid] += r

    obs = self._get_obs(obs_pids)
    rewards = {}
    for pid in obs_pids:
      rewards[pid] = self._rewards[pid]
      self._rewards[pid] = 0

    dones = {"__all__": False}
    return obs, rewards, dones, {}

  def close(self):
    if self._env:
      self._env.close()
      self._env = None
  
  def render(self):
    pass

