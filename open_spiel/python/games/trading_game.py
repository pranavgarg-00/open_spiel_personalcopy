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

_DF = pd.read_csv("/work/pranavgarg_umass_edu/open_spiel/open_spiel/python/games/aapl_data.csv")
_DONT_USE = 1000
_NUM_PLAYERS = 1
_MAX_ACTION = 10
_GAME_LENGTH = 100
_BALANCE = 1000
_RANGE = [0,5000]
_GAME_TYPE = pyspiel.GameType(
    short_name="trading_game",
    long_name="Trading Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
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
    num_distinct_actions=_MAX_ACTION*2 + 1,
    max_chance_outcomes=_RANGE[1] - _RANGE[0],
    num_players=_NUM_PLAYERS,
    min_utility=-_BALANCE,
    max_utility=_DONT_USE,
    max_game_length=_GAME_LENGTH)  


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
    self.price = -1
    self.holding = 0
    self.balance = _BALANCE
    self._rewards = np.zeros(_NUM_PLAYERS)
    self._returns = np.zeros(_NUM_PLAYERS)
    self._game_over = False
    self._next_player = 0

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
    return [a for a in range(2*_MAX_ACTION + 1) if _coord(a) >= -self.holding and _coord(a) <= self.balance//self.price[0]]


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
      self._next_player = 0
    else:
      self.holding += _coord(action)
      self.balance -= self.price[0]*_coord(action)
      self.time_elapsed += 1
      self.price = np.array(self.dataframe.iloc[self.start_time + self.time_elapsed])
      self._rewards[self._next_player] = self.holding*self.price[0] + self.balance - _BALANCE - self._returns[self._next_player]
      self._returns[self._next_player] = self.holding*self.price[0] + self.balance - _BALANCE
      self._game_over = self.time_elapsed >= _GAME_LENGTH
      self._next_player = 0

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Start Time = {action}"
    return f"Buying: {_coord(action)}"

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
                             "price:", self.price,"\n",
                             "holding:", self.holding,"\n",
                             "balance:", self.balance,"\n",
                             "_rewards", self._rewards, "\n",
                             "_returns", self._returns, "\n",
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
    pieces = []
  
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("holding", 1, (1,)))
      pieces.append(("balance", 1, (1,)))
    if iig_obs_type.public_info:
      if iig_obs_type.perfect_recall:
        pieces.append(("price_history", _GAME_LENGTH*5, (_GAME_LENGTH, 5)))
      else:
        pieces.append(("current_price", 5, (5,)))

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
    if "holding" in self.dict:
      self.dict["holding"][player] = state.holding
    if "balance" in self.dict:
      self.dict["balance"][player] = state.balance
    if "current_price" in self.dict:
      self.dict["current_price"][:] = state.price
    if "price_history" in self.dict:
      for i in range(state.time_elapsed):
        self.dict["price_history"][i] = np.array(state.dataframe.iloc[state.start_time + i])

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "holding" in self.dict:
      pieces.append(f"holding{state.holding}")
    if "balance" in self.dict:
      pieces.append(f"balance:{state.balance}")
    if "current_price" in self.dict:
      pieces.append(f"current_price:{state.price}")
    if "price_history" in self.dict:
      pieces.append(f"price_history:{np.array(state.dataframe.iloc[state.start_time:state.start_time+state.time_elapsed])}")
    return " ".join(str(p) for p in pieces)

def _coord(move):
  all_moves = list(range(-_MAX_ACTION, _MAX_ACTION + 1))
  return all_moves[move]
# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, TradingGameGame)
