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

# Base functions and training environment for RL
# Expects softmax prediction over all actions as model output

class RL_base_v2():

    def __init__(self, num_profiles):
        # Tunable learning parameters
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.exploration_rate = 0.4

        # 0 = no decay
        # 1 = decay over all profiles
        # 2 = decay per profile (doesn't work)
        # Note: If using Adam optimizer it doesn't matter
        self.f_learning_rate_decay = 1
        self.f_exploration_rate_decay = True

        self.exploration_rate_start = 0.9
        self.exploration_rate_end = 0.1
        self.exploration_rate_decay = 600000

        self.learning_rate_start = 0.9
        self.learning_rate_end = 0.05
        self.learning_rate_decay = 2000000

        # after how many iterations to update the target network to the agent's learned network
        self.update_target_network_every = 25

        self.num_iterations = 100

        self.num_profiles = num_profiles

        # 1 = eps greedy
        # 2 = boltzmann
        self.exploration_type = 2

        # used in boltzmann
        self.tau_start = 1
        self.tau_end = 0.1
        self.tau_decay = 4000000
        self.tau = self.tau_start

        # debug_mode
        # = 0: no output
        # = 1: outputs only initial state
        # = 2: outputs on stop conditions
        # = 3: outputs all data
        self.debug_mode = 0


    # one iteration of learning
    def learning_iteration(self, agent):
        # Reset environment
        agent.reset_environment()

        # While not reached goal state
        while agent.at_goal_state() == -1:
            legal_actions = agent.get_legal_actions()

            if self.debug_mode >= 2:
                agent.print_state()
                print("legal actions:", legal_actions)

            if len(legal_actions) == 0:
                # No possible actions
                if self.debug_mode >= 2:
                    print("no legal actions")
                break

            if self.f_exploration_rate_decay:
                # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                eps_threshold = self.exploration_rate_end + (self.exploration_rate_start - self.exploration_rate_end) * math.exp(
                    -1. * agent.running_nodes / self.exploration_rate_decay)
            else:
                eps_threshold = self.exploration_rate

            self.exploration_rate = eps_threshold

            if self.exploration_type == 1:
                # epsilon greedy
                if random.random() < eps_threshold:
                    # Randomly select a possible action with probability epsilon
                    i = random.randint(0, len(legal_actions) - 1)
                    a = legal_actions[i]
                    if self.debug_mode >= 2:
                        print("randomly select action", a)
                else:
                    # Otherwise greedily choose best action
                    action_Q_vals = agent.get_Q_vals()
                    max_action = None
                    max_action_val = float("-inf")

                    for e in legal_actions:
                        action_Q_val = action_Q_vals[e]
                        if action_Q_val > max_action_val:
                            max_action = e
                            max_action_val = action_Q_val

                    a = max_action
                    assert max_action is not None
                    if self.debug_mode >= 2:
                        print("greedily select action", a, "with q val", max_action_val)
            elif self.exploration_type == 2:
                # Boltzmann
                self.tau = self.tau_end + (self.tau_start - self.tau_end) * math.exp(
                    -1. * agent.running_nodes / self.tau_decay)

                action_Q_vals = agent.get_Q_vals()
                q_vals_boltz = []

                for e in legal_actions:
                    q_vals_boltz.append(exp(action_Q_vals[e].item() / self.tau))
                q_sum = sum(q_vals_boltz)
                probs = []
                for v in q_vals_boltz:
                    probs.append(v / q_sum)
                legal_actions_index = [i for i in range(len(legal_actions))]
                a = legal_actions[np.random.choice(legal_actions_index, p=probs)]

            assert (a is not None)

            # Take the action and update q vals
            self.update_q(agent, a)

            if self.f_learning_rate_decay == 1:
                # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html (originally for exploration rate decay)
                self.learning_rate = self.learning_rate_end + (self.learning_rate_start - self.learning_rate_end) * math.exp(
                    -1. * agent.running_nodes / self.learning_rate_decay)

        # Reached goal state
        agent.goal_state_update()



    '''
    Main reinforcement learning loop
    agent is the selected agent for learning
    env0 is the data given for initializing the environment (i.e. a profile)
    '''
    def reinforcement_loop(self, agent, env0, f_train_until_found_all_winners = 0, true_winners = set()):
        # Initialize
        #stats = agent.Stats()
        agent.initialize(env0)

        iter_to_find_all_winners = 0

        if f_train_until_found_all_winners:
            while agent.known_winners != true_winners:
                self.learning_iteration(agent)

                if iter_to_find_all_winners % self.update_target_network_every == 0:
                    agent.target_model.load_state_dict(agent.model.state_dict())

                iter_to_find_all_winners += 1

        for iter in range(self.num_iterations):
            self.learning_iteration(agent)

            if self.f_learning_rate_decay == 2:
                # from http://www.cs.cmu.edu/afs/andrew/course/15/381-f08/www/lectures/HandoutModelFreeRL.pdf
                # DONT USE - broken
                # TODO fix
                self.learning_rate = (self.num_iterations*self.num_profiles) / (self.num_iterations*self.num_profiles + iter * iter)

            # if self.f_exploration_rate_decay:
            #     self.exploration_rate = (self.num_iterations * self.num_profiles) / (self.num_iterations * self.num_profiles + iter * iter)

            # update target network
            if iter % self.update_target_network_every == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())

        if self.debug_mode >= 2:
            agent.print_model("")

        return agent, iter_to_find_all_winners


    def update_q(self, agent, a):
        # this works since we are backpropogating based on state and a
        # and old_q_value is computed using the model network, which gets the Variable correctly
        # and creates the gradient operation graph

        action_Q_vals = agent.get_Q_vals()

        old_q_value = action_Q_vals[a]

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

        action_next_Q_vals = agent.get_Q_vals(use_target_net=True)

        for e in next_legal_actions:
            max_next_q_val = max(max_next_q_val, action_next_Q_vals[e])

        new_q_value = new_reward + self.discount_factor * max_next_q_val

        agent.update_q(self.learning_rate, old_q_value, new_q_value)