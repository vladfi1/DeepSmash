"""All namedtuples for holding slippi data."""

from collections import namedtuple
import enum

State = namedtuple('State', 'players stage')
StateAction = namedtuple('StateAction', 'state action')

PlayerTuple = namedtuple('PlayerTuple', 'x y character action_state action_frame damage shield')

class Player(PlayerTuple):
  """For compatibility with phillip's types."""
  @property
  def percent(self):
    return self.damage

simple_buttons = "A B Z Y L DPAD_UP".split()
SimpleButtons = namedtuple("SimpleButtons", simple_buttons)

Stick = namedtuple('Stick', 'x y')

class SimpleCStick(enum.IntEnum):
  NONE = 0
  CSTICK_RIGHT = 1
  CSTICK_LEFT = 2
  CSTICK_DOWN = 3
  CSTICK_UP = 4

SimpleController = namedtuple('SimpleController', 'buttons joystick cstick trigger')
RepeatedAction = namedtuple('RepeatedAction', 'action repeat')

