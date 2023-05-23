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

"""Python Deep CFR example."""
from datetime import datetime

from absl import app
from absl import flags
from absl import logging
import random
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
import pyspiel
from open_spiel.python.pytorch import deep_cfr
import torch

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 400, "Number of iterations")
flags.DEFINE_integer("num_traversals", 40, "Number of traversals/games")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")
flags.DEFINE_float("lr", 1e-3, "learning rate")
flags.DEFINE_integer("layer_size", 16, "Size of layers")

def evaluate_random_runs(game_name, num_runs):
    game = pyspiel.load_game(FLAGS.game_name)
    run_returns_p0 = []
    run_returns_p1 = []
    for i in range(num_runs):
      run_state = game.new_initial_state()
      while not run_state.is_terminal():
        if run_state.is_chance_node():
          run_outcomes = run_state.chance_outcomes()
          run_action_list, run_prob_list = zip(*run_outcomes)
          run_chance_action = np.random.choice(run_action_list, p=run_prob_list)
          run_state.apply_action(run_chance_action)
        else:
          action = random.choice(run_state.legal_actions(run_state.current_player()))
          run_state.apply_action(action)
      run_returns_p0.append(run_state.returns()[0])
      run_returns_p1.append(run_state.returns()[1])
    return_mean_p0 = np.mean(run_returns_p0)
    return_mean_p1 = np.mean(run_returns_p1)
    return_std_p0 = np.std(run_returns_p0)
    return_std_p1 = np.std(run_returns_p1)
    return return_mean_p0, return_mean_p1, return_std_p0, return_std_p1   

def main(unused_argv):
  logging.info("Game name: " + str(FLAGS.game_name))
  logging.info("lr: " + str(FLAGS.lr))
  logging.info("layer_size: " + str(FLAGS.layer_size))
  logging.info("Loading %s" + str(FLAGS.game_name))
  game = pyspiel.load_game(FLAGS.game_name)
  retavg0, retavg1, retstd0, retstd1 = evaluate_random_runs(FLAGS.game_name, 20000)
  logging.info("\n\n")
  logging.info("RANDOM POLICY Player 0 avg returns on 20000 runs: " + str(retavg0))
  logging.info("RANDOM POLICY Player 1 avg returns on 20000 runs: " + str(retavg1))
  logging.info("RANDOM POLICY Player 0 std on 20000 runs: " + str(retstd0))
  logging.info("RANDOM POLICY Player 1 std on 20000 runs: " + str(retstd1))

  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(2*FLAGS.layer_size, 2*FLAGS.layer_size),
      advantage_network_layers=(FLAGS.layer_size, FLAGS.layer_size),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=FLAGS.lr,
      batch_size_advantage=None,
      batch_size_strategy=None,
      memory_capacity=int(1e7))

  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                 len(deep_cfr_solver.advantage_buffers[player]))
  logging.info("Strategy Buffer Size: '%s'",
               len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)


  # average_policy_values = expected_game_score.policy_value(
  #     game.new_initial_state(), [average_policy] * 2)
  # logging.info("Computed player 0 value: %.2f",
  #              average_policy_values[0])
  # logging.info("Computed player 1 value: %.2f",
  #              average_policy_values[1])
  now = datetime.now() 

  print("Game name: ", FLAGS.game_name)
  print("lr: ", FLAGS.lr)
  print("layer_size: ", FLAGS.layer_size)
  print(now.strftime("%m/%d/%Y, %H:%M:%S"))

  torch.save(deep_cfr_solver._policy_network, "/work/pranavgarg_umass_edu/deepcfr_trading_model" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".pt")

if __name__ == "__main__":
  app.run(main)
