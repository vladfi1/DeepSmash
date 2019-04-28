import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('path', type=str, help='path to compressed data')

args = parser.parse_args()

with open(args.path, 'rb') as f:
  data = pickle.load(f)

for game in data:
  game.action.repeat[:] -= 1
  assert game.action.repeat.min() == 0

with open(args.path, 'wb') as f:
  pickle.dump(data, f)

