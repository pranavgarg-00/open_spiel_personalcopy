
# OpenSpiel Fork for Stock Trading

My fork of the Open Spiel RL Library for using its algorithms in a custom stock trading environment.
The objective was to utilize deep reinforcement learning to train a fully automated stock trading agent. 
I used Deep Counterfactual Regret Minimization, an algorithm popularly used to solve poker games. 
The trained agent achieved an expected annual profit of 25% with a standard deviation of 35%. 

## File Locations:

Environment: [open_spiel/python/games/trading_game.py](open_spiel/python/games/trading_game.py)

DeefCFR agent: [open_spiel/python/examples/deep_cfr_pytorch.py](open_spiel/python/examples/deep_cfr_pytorch.py)

Trained Networks: [trading_files/\*.pt](trading_files/)

Training Logs: [trading_files/\*.out](trading_files/)

## Installation

Clone this repository and follow the "Install from Source" instructions at [docs/install.md](docs/install.md)
