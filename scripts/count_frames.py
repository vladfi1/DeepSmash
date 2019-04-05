import os
from slippi import Game

total_frames = 0

for dirpath, _, filenames in os.walk('replays/Gang-Steals/stream'):
  for fname in filenames:
    game = Game(os.path.join(dirpath, fname))
    total_frames += len(game.frames)

print("total frames", total_frames)
