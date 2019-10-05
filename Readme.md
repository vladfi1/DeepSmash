Rewrite of https://github.com/vladfi1/phillip.

# Installation

It is recommended that you first create a virtual env:
```
virtualenv venv
source venv/bin/activate
```

To install:

```
pip install -e .
```


# Imitation Learning

First download the raw .slp files into the `replays/` folder. A script is provided for the "Gang Steals" tournament.
```
./scripts/download-gang-steal.sh
```

Next, create the training data:

```
python dsmash/slippi/data.py --compress replays/Gang-Steals
```

This may take a while depending on how many replays you are processing. It will create a file called `il-data/Gang-Steals_compressed.pkl`. We can now train on this data:
```
python dsmash/imitation/train.py --data_path il-data/Gang-Steals_compressed.pkl
```

You will see `mean_action_logp` periodically printed, which measures how good your model is at predicting the actions taken by the recorded players. Higher is better, with the limit being zero.

## Running a trained model

Not implemented yet. It was implemented on the (now out of date) rllib code-path. Would be very nice to eval against a level 9 cpu while training.

