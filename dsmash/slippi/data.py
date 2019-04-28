import os
import pickle
import time

import numpy as np
from tensorflow.python.util import nest

import ubjson
import slippi
from slippi import Game, event

from dsmash.action import discrete_trigger, discrete_stick
from dsmash.slippi.types import *

Buttons = event.Buttons
Physical = Buttons.Physical
Logical = Buttons.Logical

class InvalidGameError(ValueError):
  pass


def get_simple_buttons(pre):
  physical = pre.buttons.physical
  # pause is ok because slippi doesn't record data during pause
  return SimpleButtons(
    A=bool(physical & Physical.A),
    B=bool(physical & Physical.B),
    Z=bool(physical & Physical.Z),
    Y=bool(physical & (Physical.X | Physical.Y)),
    L=bool(physical & (Physical.L | Physical.R)),
    DPAD_UP=bool(physical & Physical.DPAD_UP),
  )


def get_simple_c(pre):
  logical = pre.buttons.logical
  if logical & Logical.CSTICK_RIGHT:
    return SimpleCStick.CSTICK_RIGHT
  if logical & Logical.CSTICK_LEFT:
    return SimpleCStick.CSTICK_LEFT
  if logical & Logical.CSTICK_DOWN:
    return SimpleCStick.CSTICK_DOWN
  if logical & Logical.CSTICK_UP:
    return SimpleCStick.CSTICK_UP
  return SimpleCStick.NONE


def get_simple_controller(player, discretize):
  pre = player.pre
  id_fn = lambda x: x
  joystick_fn = discrete_stick.to_discrete if discretize else id_fn
  trigger_fn = discrete_trigger.to_discrete if discretize else id_fn
  
  return SimpleController(
      buttons=get_simple_buttons(pre),
      joystick=Stick(
          x=joystick_fn(pre.joystick.x),
          y=joystick_fn(pre.joystick.y),
      ),
      cstick=get_simple_c(pre),
      trigger=trigger_fn(pre.triggers.logical)
  )


def get_player(player):
  post = player.post
  return Player(
      x=post.position.x,
      y=post.position.y,
      character=post.character,
      action_state=post.state,
      action_frame=post.state_age,
      damage=post.damage,
      shield=post.shield,
  )

def get_state_actions(ports, start, frame, discretize=True):
  players = [get_player(frame.ports[p].leader) for p in ports]
  actions = [get_simple_controller(frame.ports[p].leader, discretize) for p in ports]

  return (
      StateAction(State(tuple(players), start.stage), actions[0]),
      StateAction(State(tuple(reversed(players)), start.stage), actions[1]),
  )

# TODO: check for other conditions?
def check_valid(game, min_len=2):
  """Checks that the game is valid for training.
  
  Current disqualifiers are:
  - teams
  - game length
  - non-human (CPU) player
  - ice climbers

  Args:
    game: The Game object.
    min_len: Minimum game length in minutes.
  Raises:
    InvalidGameError if the game is invalid.
  """
  if game.start.is_teams:
    raise InvalidGameError('teams')
  if len(game.frames) < min_len * 60 * 60:
    raise InvalidGameError('length')
  # people sometimes reset at end of game?
  # if game.end.method == event.End.Method.INCONCLUSIVE:
  #   raise InvalidGameError('inconclusive')
  for player in game.start.players:
    if player is None: continue
    if player.type != event.Start.Player.Type.HUMAN:
      raise InvalidGameError("non-human player")
    if player.character == slippi.id.CSSCharacter.ICE_CLIMBERS:
      raise InvalidGameError("ice climbers")


def get_supervised_data(game, discretize):
  player_ports = [i for i, player in enumerate(game.start.players) if player is not None]
  if len(player_ports) != 2:
    raise InvalidGameError("got %d ports" % len(player_ports))
  
  state_action_series = [get_state_actions(player_ports, game.start, f, discretize) for f in game.frames]
  for rollout in zip(*state_action_series):
    yield rollout


def compress_repeated_actions(state_actions, max_repeat=15):
  max_repeat -= 1
  repeated_state_actions = []
  last_state, last_action = state_actions[0]
  repeat = 0
  
  def commit():
    repeated_state_actions.append(
      StateAction(last_state, RepeatedAction(last_action, repeat)))

  for state, action in state_actions[1:]:
    if repeat == max_repeat or action != last_action:
      commit()
      last_state = state
      last_action = action
      repeat = 1
    else:
      repeat += 1

  commit()
  return repeated_state_actions    

def nt_to_np(nts):
  return nest.map_structure(lambda *xs: np.array(xs), *nts)

test_file = 'replays/Gang-Steals/15/Game_20190309T113739.slp'
#test_game = Game(test_file)
#print(len(test_game.frames))
#test_data = get_supervised_data(test_game)


def load_supervised_data(replay_files, discretize):
  valid_files = 0
  start_time = time.time()
  for i, f in enumerate(replay_files):
    #print('loading', f)
    try:
      game = Game(f)
    except Exception as e:
      print(f, e)
      continue
    try:
      check_valid(game)
      yield from get_supervised_data(game, discretize)
      valid_files += 1
    except InvalidGameError as e:
      print(e)
    if i % 10 == 0:
      elapsed_time = time.time() - start_time
      time_per_file = elapsed_time / (i+1)
      remaining_time = time_per_file * (len(replay_files) - i - 1)
      print('%d %d %.f %.2f %.f' % (i, valid_files, elapsed_time, time_per_file, remaining_time))

    #if elapsed_time > 30: break
  print(valid_files, len(replay_files))


def create_np_dataset(replay_path, compress=False, discretize=False):
  discretize = compress or discretize

  suffix = '_raw'
  if compress:
    suffix = '_compressed'
  elif discretize:
    suffix = '_discrete'
  suffix += '.pkl'

  save_path = replay_path.replace('replays', 'il-data', 1)
  save_path = save_path.rstrip('/') + suffix
  print('saving to', save_path)

  stream_files = []
  for dirpath, _, filenames in os.walk(replay_path):
    for fname in filenames:
      stream_files.append(os.path.join(dirpath, fname))

  rollouts = load_supervised_data(stream_files, discretize)
  if compress:
    rollouts = map(compress_repeated_actions, rollouts)
  rollouts_np = list(map(nt_to_np, rollouts))
  rollouts_np_pkl = pickle.dumps(rollouts_np)
  
  print(len(rollouts_np_pkl))
  
  save_dir = save_path.rsplit('/', 1)[0]
  os.makedirs(save_dir, exist_ok=True)
  with open(save_path, 'wb') as f:
    f.write(rollouts_np_pkl)


def compute_stats(path):
  with open(path, 'rb') as f:
    compressed_rollouts_np = pickle.load(f)
  
  total_actions = 0
  total_repeats = 0
  max_repeats = 0
  for state_actions in compressed_rollouts_np:
    repeats = state_actions.action.repeat
    total_actions += len(repeats)
    total_repeats += repeats.sum()
    max_repeats = max(repeats.max(), max_repeats)
  
  print(total_actions, total_repeats / total_actions, max_repeats)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('replay_path', type=str, help='root dir containing raw .slp replays')
  parser.add_argument('--compress', action='store_true', help='compress repeated actions')
  parser.add_argument('--discrete', action='store_true', help='discretize actions')
  parser.add_argument('--stats', action='store_true', help='compute stats instead of making dataset')
  args = parser.parse_args()

  if args.stats:
    compute_stats(args.replay_path)
  else:
    create_np_dataset(args.replay_path, compress=args.compress, discretize=args.discrete)

