# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Kuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum
import pandas as pd
import numpy as np
import random

import pyspiel

class Action(enum.IntEnum):
  BUY = 0
  HOLD = 1
  SELL = 2

_DF = pd.read_csv("/work/pranavgarg_umass_edu/open_spiel/open_spiel/python/games/aapl_data.csv")
_NUM_PLAYERS = 2
_MAX_ACTION = 3
_GAME_LENGTH = 10
_BALANCE = 0
_RANGE = [0,5000]
_GAME_TYPE = pyspiel.GameType(
    short_name="trading_game",
    long_name="Trading Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_tensor=True,
    provides_information_state_string=True,
    provides_observation_tensor=True,
    provides_observation_string=True,
    parameter_specification={
      "max_action": _MAX_ACTION,
      "game_length": _GAME_LENGTH,
      "balance": _BALANCE,
      "range_start":_RANGE[0],
      "range_end":_RANGE[1]
    })
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_MAX_ACTION,
    max_chance_outcomes=_RANGE[1] - _RANGE[0],
    num_players=_NUM_PLAYERS,
    min_utility=-50,
    max_utility=50,
    max_game_length=_GAME_LENGTH*2)  


class TradingGameGame(pyspiel.Game):
  """A Python implementation of a stock trading env"""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return TradingGameState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return TradingGameObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class TradingGameState(pyspiel.State):
  """A python implementation of a stock trading state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.dataframe = _DF
    self.start_time = -1
    self.time_elapsed = 0
    self.start_price = -1
    self.start_volume = -1
    self.price = np.array([-1,-1,-1,-1,-1])
    self.holding = [0, 0]
    self.balance = [_BALANCE, _BALANCE]
    self._game_over = False
    self._next_player = 0
    self._rewards = [0, 0]
    self._returns = [0, 0]

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self.start_time < 0:
      return pyspiel.PlayerId.CHANCE
    else:
      return self._next_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    if self.holding[player] == 0:
      return [Action.BUY, Action.HOLD]
    return [Action.HOLD, Action.SELL]


  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = list(range(_RANGE[0], _RANGE[1]))
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      self.start_time = action
      self.price = np.array(self.dataframe.iloc[self.start_time])
      self.start_price = self.price[0]
      self.start_volume = self.price[1]
      self.price = self.price/self.start_price
      self.price[1] = self.price[1]*self.start_price/self.start_volume
      self._next_player = 0
    else:
      if action==Action.BUY:
        self.holding[self._next_player] = 1
        self.balance[self._next_player] -= self.price[0]
      elif action==Action.SELL:
        self.holding[self._next_player] = 0
        self.balance[self._next_player] += self.price[0]
      if self._next_player == 1:
        self.time_elapsed += 1
      self.price = np.array(self.dataframe.iloc[self.start_time + self.time_elapsed])/self.start_price
      self.price[1] = self.price[1]*self.start_price/self.start_volume
      self._rewards = [self.holding[0]*self.price[0] + self.balance[0] - self._returns[0], self.holding[1]*self.price[0] + self.balance[1] - self._returns[1]]
      self._returns = [self._returns[0] + self._rewards[0], self._returns[1] + self._rewards[1]]
      self._game_over = self.time_elapsed >= _GAME_LENGTH
      self._next_player = 1 - self._next_player

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Start Time = {action}"
    elif action == Action.BUY:
      return "Buy"
    elif action == Action.HOLD:
      return "Hold"
    else:
      return "Sell"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def rewards(self):
    """Reward at the previous step."""
    return self._rewards

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns
  
  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "".join(map(str, ["Start Time:", self.start_time,"\n",
                             "Time Elapsed:", self.time_elapsed,"\n",
                             "start Price:", self.start_price, "\n",
                             "price:", self.price,"\n",
                             "holding:", self.holding,"\n",
                             "balance:", self.balance,"\n",
                             "_game_over:",self._game_over,"\n",
                             "_next_player",self._next_player,"\n"
                             ]))


class TradingGameObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", 1, (1,))]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("holding", 1, (1,)))
      pieces.append(("balance", 1, (1,)))
    if iig_obs_type.public_info:
      pieces.append(("time_elapsed", 1, (1,)))
      pieces.append(("price", 5, (5,)))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if "player" in self.dict:
      self.dict["player"][0] = player
    if "holding" in self.dict:
      self.dict["holding"][0] = state.holding[player]
    if "balance" in self.dict:
      self.dict["balance"][0] = state.balance[player]
    if "time_elapsed" in self.dict:
      self.dict["time_elapsed"][0] = state.time_elapsed
    if "price" in self.dict:
      self.dict["price"][:] = state.price
    

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p:{player}")
    if "holding" in self.dict:
      pieces.append(f"holding:{state.holding[player]}")
    if "balance" in self.dict:
      pieces.append(f"balance:{state.balance[player]}")
    if "time_elapsed" in self.dict:
      pieces.append(f"time_elapsed:{state.time_elapsed}")
    if "price" in self.dict:
      pieces.append(f"price:{state.price}")
    return " ".join(str(p) for p in pieces)

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, TradingGameGame)
