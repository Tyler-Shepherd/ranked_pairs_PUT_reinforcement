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

import params
import RP_utils


class Experience_Replay():
    # https://medium.com/ml-everything/reinforcement-learning-with-sparse-rewards-8f15b71d18bf
    def __init__(self, buffer_size=50000, unusual_sample_factor=0.99):
        """ Data structure used to hold game experiences """
        # Buffer will contain [state,action,reward,next_state,done]
        self.buffer = []
        self.buffer_size = buffer_size

        assert unusual_sample_factor <= 1, "unusual_sample_factor has to be <= 1"
        # Setting this value to a low number over-samples experience that had unusually high or low rewards
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
        self.learning_rate = params.learning_rate
        self.exploration_rate = params.exploration_rate
        self.tau = params.tau_start

        self.num_profiles = num_profiles

        self.num_times_trained = 0

        self.buffer = Experience_Replay(buffer_size=params.buffer_size, unusual_sample_factor=params.unusual_sample_factor)

    '''
    Performs one iteration of learning
    An "iteration" is one full run of RP from initial state to goal state
    If full_K is 1 then sets agents K to be known_winners (used when training to find all winners)
    iter_to_find_winner is, if defined, a dict of winner to number of iterations needed to discover that winner
    Does not sample from experience replay buffer
    '''
    def learning_iteration(self, agent, full_K = 0, iter_to_find_winner = None):
        # Reset environment
        agent.reset_environment(iter_to_find_winner = iter_to_find_winner)

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

        # Reached goal state
        agent.goal_state_update()


    '''
    Main reinforcement learning loop
    agent is the selected agent for learning
    env0 is the data given for initializing the environment (i.e. a profile)
    f_train_until_found_all_winners: if true, will continue training until all winners are found (specified by true_winners) then do num_training_iterations
    '''
    def reinforcement_loop(self, agent, env0, f_train_until_found_all_winners = 0, true_winners = set()):
        # Initialize
        agent.initialize(env0)

        iter_to_find_all_winners = 0

        iter_to_find_winner = {}
        prev_winners = set()

        if f_train_until_found_all_winners:
            # Accumulates experiences without training
            while agent.known_winners != true_winners and iter_to_find_all_winners < params.cutoff_training_iterations:
                assert agent.known_winners < true_winners

                self.learning_iteration(agent, full_K=1)

                iter_to_find_all_winners += 1

                for c in agent.known_winners - prev_winners:
                    iter_to_find_winner[c] = iter_to_find_all_winners
                prev_winners = agent.known_winners.copy()

        for iter in range(params.num_training_iterations):
            self.learning_iteration(agent, iter_to_find_winner=iter_to_find_winner)

            # Train from samples of experience replay buffer
            if (iter + 1) % params.train_from_experiences_every_iterations == 0:
                # print("Training iter", iter)
                self.train(agent)

                # Reinitialize to the currently training profile
                agent.initialize(env0)

        return agent, iter_to_find_winner, iter_to_find_all_winners


    '''
    Agent takes action a
    Store experience in buffer
    '''
    def update_q(self, agent, a):
        # Experience of form [state,action,reward,next_state,done]
        experience = []
        experience.append(agent.get_current_state())
        experience.append(a)

        # Actually updates the agent state
        agent.make_move(a, f_testing=True)
        agent.running_nodes += 1

        # Gets reward of current (now updated) state
        new_reward = agent.reward()
        experience.append(new_reward)
        experience.append(agent.get_current_state())
        experience.append(agent.at_goal_state() != -1)

        self.buffer.add([experience])


    '''
    Samples from experience replay buffer to train agent
    '''
    def train(self, agent):
        if len(self.buffer.buffer) == 0:
            # Nothing in buffer, nothing to train
            return
        samples = self.buffer.sample(params.batch_size)

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
                max_next_q_val = max(max_next_q_val, agent.get_Q_val(e, use_target_net=True))
            new_q_value = reward + params.discount_factor * max_next_q_val

            agent.update_q(self.learning_rate, old_q_value, new_q_value)

            if params.f_learning_rate_decay == 1:
                # from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html (originally for exploration rate decay)
                self.learning_rate = params.learning_rate_end + (params.learning_rate_start - params.learning_rate_end) * math.exp(-1. * agent.running_nodes / params.learning_rate_decay)

        # avg loss per sample
        loss_avg = agent.running_loss / params.batch_size
        print(self.num_times_trained, ":*******LOSS:", loss_avg)
        if agent.loss_output_file:
            agent.loss_output_file.write(str(self.num_times_trained) + '\t' + str(loss_avg) + '\n')
            agent.loss_output_file.flush()

        agent.running_loss = 0
        self.num_times_trained += 1

        # update target network
        if self.num_times_trained % params.update_target_network_every == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
