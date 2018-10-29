import io
import os
import prefpy_io
import math
import time
from numpy import *
import numpy as np
import itertools
from preference import Preference
from profile import Profile
import copy
import sys
import networkx as nx
import rpconfig
from collections import defaultdict
import matplotlib.pyplot as plt
from queue import PriorityQueue
import random
from pprint import pprint
import glob

import params as params

# Base functions and training environment for RL

class RL_base():

    def __init__(self, num_profiles):
        self.learning_rate = params.learning_rate
        self.exploration_rate = params.exploration_rate
        self.tau = params.tau_start

        self.num_profiles = num_profiles

        self.distribution_output_file = open('winners_distribution.txt', 'w+')

    '''
    Performs one iteration of learning
    An "iteration" is one full run of RP from initial state to goal state
    If full_K is 1 then sets agents K to be known_winners (used when training to find all winners)
    iter_to_find_winner is, if defined, a dict of winner to number of iterations needed to discover that winner
    '''
    def learning_iteration(self, agent, full_K = 0, iter_to_find_winner = None, winners_distribution = None):
        # Reset environment
        reset_success = agent.reset_environment(iter_to_find_winner = iter_to_find_winner, winners_distribution = winners_distribution)
        if reset_success == -1:
            return

        if full_K:
            agent.K = frozenset(agent.known_winners)

        # While not reached goal state
        while agent.at_goal_state()[0] == -1:
            legal_actions = agent.get_legal_actions()

            if params.debug_mode >= 2:
                agent.print_state()
                print("legal actions:", legal_actions)

            if params.debug_mode >= 3:
                for e in legal_actions:
                    print("action, q val", e, agent.get_Q_val(e).item())

            if len(legal_actions) == 0:
                # No possible actions
                if params.debug_mode >= 2:
                    print("no legal actions")
                break

            # Explore/exploit
            if params.exploration_algo == 1:
                # epsilon greedy
                if params.f_exploration_rate_decay:
                    # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                    eps_threshold = params.exploration_rate_end + (params.exploration_rate_start - params.exploration_rate_end) * math.exp(
                        -1. * agent.running_nodes / params.exploration_rate_decay)
                else:
                    eps_threshold = self.exploration_rate

                self.exploration_rate = eps_threshold

                if random.random() < eps_threshold:
                    # Randomly select a possible action with probability epsilon
                    i = random.randint(0, len(legal_actions) - 1)
                    a = legal_actions[i]
                    if params.debug_mode >= 2:
                        print("randomly select action", a)
                else:
                    # Otherwise greedily choose best action
                    max_action = None
                    max_action_val = float("-inf")
                    for e in legal_actions:
                        action_Q_val = agent.get_Q_val(e)
                        if action_Q_val > max_action_val:
                            max_action = e
                            max_action_val = action_Q_val

                    a = max_action
                    if params.debug_mode >= 2:
                        print("greedily select action", a, "with q val", max_action_val)
            elif params.exploration_algo == 2:
                # Boltzmann
                q_vals = []
                self.tau = params.tau_end + (params.tau_start - params.tau_end) * math.exp(
                    -1. * agent.running_nodes / params.tau_decay)
                for e in legal_actions:
                    q_vals.append(exp(agent.get_Q_val(e).item() / self.tau))
                q_sum = sum(q_vals)
                probs = []
                for v in q_vals:
                    probs.append(v / q_sum)
                legal_actions_index = [i for i in range(len(legal_actions))]
                a = legal_actions[np.random.choice(legal_actions_index, p=probs)]

            assert a is not None

            # Take the action and update q vals
            self.update_q(agent, a)

            if params.f_learning_rate_decay == 1:
                # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html (originally for exploration rate decay)
                self.learning_rate = params.learning_rate_end + (params.learning_rate_start - params.learning_rate_end) * math.exp(
                    -1. * agent.running_nodes / params.learning_rate_decay)

        # Reached goal state
        agent.goal_state_update()


    '''
    Main reinforcement learning loop
    agent is the selected agent for learning
    env0 is the data given for initializing the environment (i.e. a profile)
    f_train_until_found_all_winners: if true, will continue training until all winners are found (specified by true_winners) then do num_training_iterations
    '''
    def reinforcement_loop(self, agent, env0, true_winners = set(), filename = None, winners_distribution = None):
        # Initialize
        agent.initialize(env0)

        iter_to_find_all_winners = 0

        iter_to_find_winner = {}
        prev_winners = set()

        # for getting winners distribution data
        start = time.perf_counter()
        winner_to_num_times_found = {}
        while iter_to_find_all_winners < 1000:
            assert agent.known_winners < true_winners

            agent.reset_environment()
            agent.K = frozenset()

            # While not reached goal state
            while agent.at_goal_state()[0] == -1:
                legal_actions = agent.get_legal_actions()
                a = legal_actions[random.randint(0, len(legal_actions) - 1)]
                agent.make_move(a, f_testing = True)

            current_state, possible_winners = agent.at_goal_state(update_stats=0)

            assert current_state == 1 or current_state == 3

            for c in possible_winners:
                if c not in winner_to_num_times_found:
                    winner_to_num_times_found[c] = 0
                winner_to_num_times_found[c] += 1

            iter_to_find_all_winners += 1

        print(filename, winner_to_num_times_found)
        self.distribution_output_file.write(str(filename) + '\n')
        for c in true_winners:
            if c in winner_to_num_times_found:
                self.distribution_output_file.write(str(c) + '\t' + str(winner_to_num_times_found[c]) + '\n')
            else:
                self.distribution_output_file.write(str(c) + '\t' + str(0) + '\n')

        self.distribution_output_file.flush()
        print(time.perf_counter() - start)

        return agent, 0, 0

        if params.f_train_till_find_all_winners:
            while agent.known_winners != true_winners and iter_to_find_all_winners < params.cutoff_training_iterations:
                assert agent.known_winners < true_winners

                self.learning_iteration(agent, full_K=1)

                if iter_to_find_all_winners % params.update_target_network_every == 0:
                    agent.target_model.load_state_dict(agent.model.state_dict())

                iter_to_find_all_winners += 1

                for c in agent.known_winners - prev_winners:
                    iter_to_find_winner[c] = iter_to_find_all_winners
                prev_winners = agent.known_winners.copy()


        for iter in range(params.num_training_iterations):
            if params.f_use_winners_distribution:
                self.learning_iteration(agent, winners_distribution = winners_distribution)
            elif params.f_train_till_find_all_winners:
                self.learning_iteration(agent, iter_to_find_winner=iter_to_find_winner)
            else:
                self.learning_iteration(agent)

            # if params.f_learning_rate_decay == 2:
            #     # from http://www.cs.cmu.edu/afs/andrew/course/15/381-f08/www/lectures/HandoutModelFreeRL.pdf
            #     # DONT USE - broken
            #     # TODO fix
            #     assert 2 == 1
            #     self.learning_rate = (self.num_training_iterations*self.num_profiles) / (self.num_training_iterations*self.num_profiles + iter * iter)

            # update target network
            if iter % params.update_target_network_every == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

        if params.debug_mode >= 2:
            agent.print_model("")

        return agent, iter_to_find_winner, iter_to_find_all_winners

    '''
    Takes action a and updates agent q values 
    '''
    def update_q(self, agent, a):
        # this works since we are backpropogating based on state and a
        # and old_q_value is computed using the model network, which gets the Variable correctly
        # and creates the gradient operation graph
        old_q_value = agent.get_Q_val(a)

        # Actually updates the agent state
        agent.make_move(a)

        # Gets reward of current (now updated) state
        new_reward = agent.reward()

        # Get the maximum estimated q value of all possible actions after adding a
        max_next_q_val = float("-inf")
        next_legal_actions = agent.get_legal_actions()

        if len(next_legal_actions) == 0:
            # If there are no legal next actions, then we've reached a goal state
            # Estimate of next state is just 0 (since there is no next state)
            max_next_q_val = 0

        for e in next_legal_actions:
            max_next_q_val = max(max_next_q_val, agent.get_Q_val(e, use_target_net=True))

        new_q_value = new_reward + params.discount_factor * max_next_q_val

        agent.update_q(self.learning_rate, old_q_value, new_q_value)