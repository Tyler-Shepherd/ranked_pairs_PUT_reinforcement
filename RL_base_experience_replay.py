import io
import os
import prefpy_io
import math
import time
import numpy as np
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


class Experience_Replay():
    # https://medium.com/ml-everything/reinforcement-learning-with-sparse-rewards-8f15b71d18bf
    def __init__(self, buffer_size=50000, unusual_sample_factor=0.99):
        """ Data structure used to hold game experiences """
        # Buffer will contain [state,action,reward,next_state,done]
        self.buffer = []
        self.buffer_size = buffer_size

        assert unusual_sample_factor <= 1, "unusual_sample_factor has to be <= 1"
        # Setting this value to a low number over-samples experience that had unusually high or
        # low rewards
        self.unusual_sample_factor = unusual_sample_factor

    '''
    param experience is a list of experiences (where each experience is a list of form [state,action,reward,next_state,done]
    '''
    def add(self, experience):
        """ Adds list of experiences to the buffer """
        # Extend the stored experiences
        self.buffer.extend(experience)
        # Keep the last buffer_size number of experiences
        self.buffer = self.buffer[-self.buffer_size:]
        # Keep the extreme values near the top of the buffer for oversampling

    def sample(self, size):
        """ Returns a sample of experiences from the buffer """
        # We want to over-sample frames where things happened. So we'll sort the buffer on the absolute reward
        # (either positive or negative) and apply a geometric probability in order to bias our sampling to the
        # earlier (more extreme) replays
        buffer = sorted(self.buffer, key=lambda replay: abs(replay[2]), reverse=True)
        p = np.array([self.unusual_sample_factor ** i for i in range(len(buffer))])
        p = p / sum(p)
        sample_idxs = np.random.choice(np.arange(len(buffer)), size=size, p=p)
        sample_output = [buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output, (size, -1))
        return sample_output


# Base functions and training environment for RL
class RL_base_experience_replay():

    def __init__(self, num_profiles):
        # Tunable learning parameters
        self.learning_rate = 0.04
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
        self.exploration_rate_decay = 20000

        self.learning_rate_start = 0.9
        self.learning_rate_end = 0.05
        self.learning_rate_decay = 2000

        self.num_iterations = 1000

        self.num_profiles = num_profiles

        self.num_times_trained = 0
        self.train_every_iterations = 200

        # debug_mode
        # = 0: no output
        # = 1: outputs only initial state
        # = 2: outputs on stop conditions
        # = 3: outputs all data
        self.debug_mode = 0

        self.buffer_size = 50000
        self.unusual_sample_factor = 0.99
        self.batch_size = 100

        self.buffer = Experience_Replay(buffer_size=self.buffer_size, unusual_sample_factor=self.unusual_sample_factor)


    '''
    Main reinforcement learning loop
    agent is the selected agent for learning
    env0 is the data given for initializing the environment (i.e. a profile)
    '''
    def reinforcement_loop(self, agent, env0):
        # Initialize
        agent.initialize(env0)

        for iter in range(self.num_iterations):
            # Reset environment
            agent.reset_environment()

            # While not reached goal state
            while agent.at_goal_state() == -1:
                legal_actions = agent.get_legal_actions()

                if self.debug_mode >= 2:
                    print("start")
                    agent.print_state()
                    print("legal actions:", legal_actions)

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
                    if self.debug_mode >= 2:
                        print("greedily select action", a, "with q val", max_action_val)

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

            if iter % self.train_every_iterations == 0:
                self.train(agent)

            # if self.f_exploration_rate_decay:
            #     self.exploration_rate = (self.num_iterations * self.num_profiles) / (self.num_iterations * self.num_profiles + iter * iter)

        if self.debug_mode >= 2:
            agent.print_model()

        return agent


    '''
    Agent takes action a
    Store experience in buffer
    '''
    def update_q(self, agent, a):
        # Experience of form [state,action,reward,next_state,done]
        experience = []
        experience.append(agent.get_current_state())
        experience.append(a)

        # old_q_value = agent.get_Q_val(a)

        # Actually updates the agent state
        agent.make_move(a, f_testing=True)

        # Gets reward of current (now updated) state
        new_reward = agent.reward()
        experience.append(new_reward)
        experience.append(agent.get_current_state())
        experience.append(agent.at_goal_state() != -1)

        # Get the maximum estimated q value of all possible actions after adding a
        # max_next_q_val = float("-inf")
        # next_legal_actions = agent.get_legal_actions()
        # if len(next_legal_actions) == 0:
        #     # If there are no legal next actions, then we've reached a goal state
        #     # Estimate of next state is just 0 (since there is no next state)
        #     max_next_q_val = 0
        # for e in next_legal_actions:
        #     max_next_q_val = max(max_next_q_val, agent.get_Q_val(e))
        # new_q_value = new_reward + self.discount_factor * max_next_q_val
        # agent.update_q(self.learning_rate, old_q_value, new_q_value)

        self.buffer.add([experience])


    '''
    Samples from experience replay buffer to train agent
    '''
    def train(self, agent):
        samples = self.buffer.sample(self.batch_size)

        for s in samples:
            #[state, action, reward, next_state, done]
            state = s[0]
            action = s[1]
            reward = s[2]
            next_state = s[3]
            done = s[4]

            agent.set_state(state)

            old_q_value = agent.get_Q_val(action)

            agent.set_state(next_state)

            # Get the maximum estimated q value of all possible actions after adding a
            max_next_q_val = float("-inf")
            next_legal_actions = agent.get_legal_actions()
            if len(next_legal_actions) == 0:
                # If there are no legal next actions, then we've reached a goal state
                # Estimate of next state is just 0 (since there is no next state)
                max_next_q_val = 0
            for e in next_legal_actions:
                max_next_q_val = max(max_next_q_val, agent.get_Q_val(e))
            new_q_value = reward + self.discount_factor * max_next_q_val

            agent.update_q(self.learning_rate, old_q_value, new_q_value)

        print(self.num_times_trained, ":*******LOSS:", agent.running_loss)
        if agent.loss_output_file:
            agent.loss_output_file.write(str(self.num_times_trained) + '\t' + str(agent.running_loss) + '\n')
            agent.loss_output_file.flush()

        agent.running_loss = 0
        self.num_times_trained += 1