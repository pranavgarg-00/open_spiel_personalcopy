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
"""Tests for Python Kuhn Poker."""

from absl.testing import absltest
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms.get_all_states import get_all_states
from open_spiel.python.games import trading_game  # pylint: disable=unused-import
from open_spiel.python.observation import make_observation
import pyspiel


class TradingGameTest(absltest.TestCase):

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("trading_game")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)
 


if __name__ == "__main__":
  absltest.main()
