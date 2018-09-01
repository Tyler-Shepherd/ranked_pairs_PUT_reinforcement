import io
import os
import prefpy_io
import math
import time
from numpy import *
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
import tensorflow as tf
import random
from pprint import pprint
import glob

# Base functions and training environment for RL

class RL_base():

    def __init__(self, num_profiles):
        # Tunable learning parameters
        self.learning_rate = 0.05
        # self.learning_rate = 1.0
        self.discount_factor = 0.9
        self.exploration_rate = 0.1

        # 0 = no decay
        # 1 = decay over all profiles
        # 2 = decay per profile
        self.f_learning_rate_decay = 1
        self.f_exploration_rate_decay = True

        self.exploration_rate_start = 0.9
        self.exploration_rate_end = 0.05
        self.exploration_rate_decay = 6000000

        self.learning_rate_start = 0.9
        self.learning_rate_end = 0.05
        self.learning_rate_decay = 2000000

        self.num_iterations = 100

        self.num_profiles = num_profiles

        # debug_mode
        # = 0: no output
        # = 1: outputs only initial state
        # = 2: outputs on stop conditions
        # = 3: outputs all data
        self.debug_mode = 0


    '''
    Main reinforcement learning loop
    agent is the selected agent for learning
    env0 is the data given for initializing the environment (i.e. a profile)
    '''
    def reinforcement_loop(self, agent, env0):
        # Initialize
        #stats = agent.Stats()
        agent.initialize(env0)

        for iter in range(self.num_iterations):
            # Reset environment
            agent.reset_environment()

            # While not reached goal state
            while agent.at_goal_state() == -1:
                legal_actions = agent.get_legal_actions()

                if self.debug_mode >= 2:
                    agent.print_state()
                    print("legal actions:", legal_actions)

                if self.debug_mode >= 3:
                    for e in legal_actions:
                        print("action, q val", e, agent.get_Q_val(e).item())

                if len(legal_actions) == 0:
                    # No possible actions
                    if self.debug_mode >= 2:
                        print("no legal actions")
                    break

                if self.f_exploration_rate_decay:
                    # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                    eps_threshold = self.exploration_rate_end + (self.exploration_rate_start - self.exploration_rate_end) * math.exp(-1. * agent.running_nodes / self.exploration_rate_decay)
                else:
                    eps_threshold = self.exploration_rate

                self.exploration_rate = eps_threshold

                if random.random() < eps_threshold:
                    # Randomly select a possible action with probability epsilon
                    i = random.randint(0, len(legal_actions) - 1)
                    a = legal_actions[i]
                    if self.debug_mode >= 2:
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
                    if max_action == None:
                        print('None?')
                    if self.debug_mode >= 2:
                        print("greedily select action", a, "with q val", max_action_val)

                assert(a is not None)

                # Take the action and update q vals
                self.update_q(agent, a)

            # Reached goal state
            agent.goal_state_update()

            if self.f_learning_rate_decay == 1:
                # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html (originally for exploration rate decay)
                self.learning_rate = self.learning_rate_end + (self.learning_rate_start - self.learning_rate_end) * math.exp(
                    -1. * agent.running_nodes / self.learning_rate_decay)
            elif self.f_learning_rate_decay == 2:
                # from http://www.cs.cmu.edu/afs/andrew/course/15/381-f08/www/lectures/HandoutModelFreeRL.pdf
                # DONT USE - broken
                # TODO fix
                self.learning_rate = (self.num_iterations*self.num_profiles) / (self.num_iterations*self.num_profiles + iter * iter)

            # if self.f_exploration_rate_decay:
            #     self.exploration_rate = (self.num_iterations * self.num_profiles) / (self.num_iterations * self.num_profiles + iter * iter)

        if self.debug_mode >= 2:
            agent.print_model("")

        return agent  #, stats


    def update_q(self, agent, a):
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
            max_next_q_val = max(max_next_q_val, agent.get_Q_val(e))

        new_q_value = new_reward + self.discount_factor * max_next_q_val

        agent.update_q(self.learning_rate, old_q_value, new_q_value)