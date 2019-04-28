import enum
import os
from threading import Thread
from dsmash import util, ssbm
from dsmash.slippi import types as slp

@enum.unique
class Button(enum.Enum):
  A = 0
  B = 1
  X = 2
  Y = 3
  Z = 4
  START = 5
  L = 6
  R = 7
  D_UP = 8
  D_DOWN = 9
  D_LEFT = 10
  D_RIGHT = 11

def slp_to_dol_btn(button):
  return button.replace('DPAD', 'D')

slippi_buttons = list(map(slp_to_dol_btn, slp.simple_buttons))

@enum.unique
class Trigger(enum.Enum):
  L = 0
  R = 1

@enum.unique
class Stick(enum.Enum):
  MAIN = 0
  C = 1

def slp_to_dol_xy(x):
  return 0.5 * (x + 1)

SLP_TO_DOL_C = [
    (0.5, 0.5),  # NONE
    (1, 0.5), # RIGHT
    (0, 0.5), # LEFT
    (0.5, 0), # DOWN
    (0.5, 1), # UP
]

def name_to_enum_map(enum_class):
  return {x.name: x for x in enum_class}

# only using button
NAME_TO_BUTTON, NAME_TO_TRIGGER, NAME_TO_STICK = map(
    name_to_enum_map, [Button, Trigger, Stick])

class Pad:
  """Writes out controller inputs."""
  def __init__(self, path, tcp=False):
    """Creates the fifo or tcp socket, but doesn't block.
    Args:
      path: Path to pipe file.
      tcp: Whether to use zmq over tcp or a fifo. If true, the pipe file
      is simply a text file containing the port number. The port will
      be a hash of the path.
    """
    self.tcp = tcp
    self.path = path
    if tcp:
      import zmq
      context = zmq.Context()
      self.port = util.port(path)
      
      with open(path, 'w') as f:
        f.write(str(self.port))
    else:
      os.mkfifo(path)
    
    self.message = ""

  def connect(self):
    """Binds to the socket/fifo, blocking until dolphin is listening."""
    if self.tcp:
      self.socket = context.socket(zmq.PUSH)
      address = "tcp://127.0.0.1:%d" % self.port
      print("Binding pad %s to address %s" % (self.path, address))
      self.socket.bind(address)
    else:
      self.pipe = open(self.path, 'w', buffering=1)

  def __del__(self):
    """Closes the fifo."""
    if not self.tcp:
      self.pipe.close()
  
  def write(self, command, buffering=False):
    self.message += command + '\n'
    
    if not buffering:
      self.flush()
  
  def flush(self):
    if self.tcp:
      #print("sent message", self.message)
      self.socket.send_string(self.message)
    else:
      self.pipe.write(self.message)
    self.message = ""

  def press_button(self, button, buffering=False):
    """Press a button."""
    assert button in Button
    self.write('PRESS {}'.format(button.name), buffering)

  def release_button(self, button, buffering=False):
    """Release a button."""
    assert button in Button
    self.write('RELEASE {}'.format(button.name), buffering)

  def press_trigger(self, trigger, amount, buffering=False):
    """Press a trigger. Amount is in [0, 1], with 0 as released."""
    assert trigger in Trigger
    # assert 0 <= amount <= 1
    self.write('SET {} {:.2f}'.format(trigger.name, amount), buffering)

  def tilt_stick(self, stick, x, y, buffering=False):
    """Tilt a stick. x and y are in [0, 1], with 0.5 as neutral."""
    assert stick in Stick
    try:
      assert 0 <= x <= 1 and 0 <= y <= 1
    except AssertionError:
      import ipdb; ipdb.set_trace()
    self.write('SET {} {:.2f} {:.2f}'.format(stick.name, x, y), buffering)

  def send_controller(self, controller):
    if isinstance(controller, ssbm.RealControllerState):
      self.send_real_controller(controller)
    elif isinstance(controller, slp.SimpleController):
      self.send_simple_controller(controller)
    else:
      raise TypeError('Unknown controller type %s' % type(controller))

  def send_real_controller(self, controller):
    """Sends a dsmash.ssbm.RealControllerState."""
    for button in Button:
      field = 'button_' + button.name
      if hasattr(controller, field):
        if getattr(controller, field):
          self.press_button(button, True)
        else:
          self.release_button(button, True)

    # for trigger in Trigger:
    #   field = 'trigger_' + trigger.name
    #   self.press_trigger(trigger, getattr(controller, field))

    for stick in Stick:
      field = 'stick_' + stick.name
      value = getattr(controller, field)
      self.tilt_stick(stick, value.x, value.y, True)
    
    self.flush()
  
  def send_simple_controller(self, controller):
    """Sends a dsmash.slippi.types.SimpleController."""
    for name, value in zip(slippi_buttons, controller.buttons):
      button = NAME_TO_BUTTON[name]  # TODO: shortcut this
      if value:
        self.press_button(button, True)
      else:
        self.release_button(button, True)

    self.tilt_stick(
        Stick.MAIN,
        slp_to_dol_xy(controller.joystick.x),
        slp_to_dol_xy(controller.joystick.y),
        True)
    
    c_x, c_y = SLP_TO_DOL_C[controller.cstick]
    self.tilt_stick(Stick.C, c_x, c_y, True)
    
    self.press_trigger(Trigger.L, controller.trigger, True)
    self.flush()

