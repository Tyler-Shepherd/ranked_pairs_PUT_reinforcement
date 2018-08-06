import io
import os
import prefpy_io
import math
import time
import numpy as np
from numpy import *
from numpy import linalg as LA
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
import torch
from torch.autograd import Variable
import random
from pprint import pprint
import glob

sys.path.append('..//..//node2vec-master//src')
import node2vec
import main as node2vecmain


class RP_RL_stats():
    def __init__(self):
        self.num_nodes = 0
        self.stop_condition_hits = {1: 0, 2: 0, 3: 0}
        self.loss = []
        self.features_time = 0


class RP_RL_agent_node2vec():
    def __init__(self):
        # Initialize learning model

        self.D_in = 258  # input dimension, node2vec as features
        self.H = 128  # hidden dimension
        self.D_out = 1  # output dimension, just want q value

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.H, self.D_out)
        )

        self.loss_fn = torch.nn.MSELoss(size_average=False)  # using mean squared error

        self.stats = RP_RL_stats()

        self.node2vec_args = node2vecmain.parse_args()
        self.node2vec_args.directed = True
        self.node2vec_args.weighted = False
        self.node2vec_args.unweighted = True
        self.node2vec_args.undirected = False
        self.node2vec_args.output = "node2vec_output.emb"

        assert 1 == 2, "This agent is broken - just update RP_RL_agent"

    '''
    Initializes environment for an iteration of learning
    env0 is a profile
    creates G_0 = initial state of RP graph (no edges)
    creates E_0 = initial state of unadded edges (all of them)
    '''
    def initialize(self, env0):
        wmg = env0.getWmg()
        self.known_winners = set()
        self.I = list(wmg.keys())

        self.G_0 = nx.DiGraph()
        self.G_0.add_nodes_from(self.I)

        self.E_0 = nx.DiGraph()
        self.E_0.add_nodes_from(self.I)
        for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
            if wmg[cand1][cand2] > 0:
                self.E_0.add_edge(cand1, cand2, weight=wmg[cand1][cand2])

        node2vec_G = node2vec.Graph(self.E_0, True, self.node2vec_args.p, self.node2vec_args.q)
        node2vec_G.preprocess_transition_probs()
        walks = node2vec_G.simulate_walks(self.node2vec_args.num_walks, self.node2vec_args.walk_length)
        self.node2vec_model = node2vecmain.learn_embeddings(walks, self.node2vec_args)

        self.stats = RP_RL_stats()

    '''
    Resets G (the current RP graph), E (the graph of unadded edges) and K (the known winners)
    '''
    def reset_environment(self):
        self.G = self.G_0.copy()
        self.E = self.E_0.copy()

        # Randomly initialize known winners
        self.K = set()
        for a in self.known_winners:
            if random.random() > 0.4:
                self.K.add(a)
        self.K = frozenset(self.K)

    '''
    Returns -1 if not at goal state
    Returns 1 if E has no more edges
    Returns 2 if pruned
    Returns 3 if only one possible winner
    '''
    def at_goal_state(self):
        in_deg = self.G.in_degree(self.I)
        possible_winners = [x[0] for x in in_deg if x[1] == 0]

        # Stop Condition 2: Pruning. Possible winners are subset of known winners
        if set(possible_winners) <= self.known_winners:
            self.stats.stop_condition_hits[2] += 1
            return 2

        # Stop Condition 1: E has no more edges
        if len(self.E.edges()) == 0:
            self.stats.stop_condition_hits[1] += 1
            return 1

        # Stop Condition 3: Exactly one node has indegree 0
        if len(possible_winners) == 1:
            self.stats.stop_condition_hits[3] += 1
            return 3

        return -1

    '''
    Returns the highest weight edges in E that do not cause a cycle if added
    '''
    def get_legal_actions(self):
        if len(self.E.edges()) == 0:
            # There are no possible actions
            return []

        Gc = self.G.copy()
        G_transitive_closure = nx.transitive_closure(self.G)

        max_weight = max([(d['weight']) for (u, v, d) in self.E.edges(data=True)])
        T = [(u, v) for (u, v, d) in self.E.edges(data=True) if d['weight'] == max_weight]

        legal_actions = []

        for e in T:
            if not G_transitive_closure.has_edge(e[1], e[0]):
                Gc.add_edges_from([e])
                legal_actions.append(e)
                Gc.remove_edges_from([e])

        return legal_actions


    def print_state(self):
        print("G:", self.G.edges())
        print("E:", self.E.edges())
        print("K:", self.K)

    # Returns input layer features at current state taking action a
    # a is an edge
    def state_features(self, a):

        start = time.perf_counter()
        #
        # G_with_weights = nx.DiGraph()
        # G_with_weights.add_nodes_from(self.I)
        #
        # for (cand1, cand2) in self.G.edges():
        #     G_with_weights.add_edge(cand1, cand2, weight=self.E_0[cand1][cand2]['weight'])
        #
        # node2vec_G = node2vec.Graph(G_with_weights, True, self.node2vec_args.p, self.node2vec_args.q)
        # node2vec_G.preprocess_transition_probs()
        # walks = node2vec_G.simulate_walks(self.node2vec_args.num_walks, self.node2vec_args.walk_length)
        # node2vec_model = node2vecmain.learn_embeddings(walks, self.node2vec_args)

        node2vec_u = self.node2vec_model.wv[str(a[0])]
        node2vec_v = self.node2vec_model.wv[str(a[1])]
        node2vec_uv = np.append(node2vec_u, node2vec_v)

        kvec = np.array([2 * int(a[0] in self.K) - 1, 2 * int(a[1] in self.K) - 1])

        node2vec_f = np.append(node2vec_uv, kvec)

        self.stats.features_time += (time.perf_counter() - start)

        return Variable(torch.from_numpy(node2vec_f).float())

    def get_Q_val(self, a):
        state_features = self.state_features(a)
        return self.model(state_features)

    # Adds new PUT-winners to self.known_winners
    def goal_state_update(self):
        G_in_degree = self.G.in_degree(self.I)
        to_be_added = set([x[0] for x in G_in_degree if x[1] == 0])

        for c in to_be_added:
            if c not in self.known_winners:
                self.known_winners.add(c)

    def print_model(self, output_file):
        for p in self.model.parameters():
            if output_file != "":
                data = p.data.numpy().tolist()
                for x in range(len(data)):
                    if isinstance(data[x],list) :
                        for j in range(len(data[x])):
                            output_file.write(str(data[x][j]) + " ")
                        output_file.write('\n')
                    else:
                        output_file.write(str(data[x]) + "\n")

            print(p)
        if output_file != "":
            output_file.write('\n')
            output_file.flush()

    '''
    Adds edge a from E to G
    '''
    def make_move(self, a):
        self.G.add_edges_from([a])
        self.E.remove_edges_from([a])

        # Remove inconsistent edges
        G_transitive_closure = nx.transitive_closure(self.G)
        Ec = self.E.copy().edges()
        for e in Ec:
            if G_transitive_closure.has_edge(e[1], e[0]):
                self.E.remove_edges_from([e])

        self.stats.num_nodes += 1

    def reward(self):
        current_state = self.at_goal_state()

        if current_state == -1:
            # Not a goal state
            reward_val =  0
        elif current_state == 2:
            # Pruning state
            reward_val = -1
        else:
            # Found a new winner
            reward_val = 1

        return torch.tensor(reward_val, dtype = torch.float32)

    def update_q(self, learning_rate, old_q_value, new_q_value):

        new_q_value = Variable(new_q_value)

        #new_q_value = Variable(torch.from_numpy(np.array([new_q_value])).float())

        # Compute loss
        loss = self.loss_fn(old_q_value, new_q_value)

        self.stats.loss.append(loss)

        # Zero the gradients before running the backward pass.
        self.model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor
        with torch.no_grad():
            for param in self.model.parameters():
                param -= learning_rate * param.grad


    def test_model(self, test_env, num_iterations):
        # Save the model
        torch.save(self.model.state_dict(), "checkpoint.pth.tar")

        self.initialize(test_env)

        # Sample using model greedily
        for iter in range(num_iterations):
            self.reset_environment()
            self.K = frozenset(self.known_winners)

            while self.at_goal_state() == -1:
                legal_actions = self.get_legal_actions()

                if len(legal_actions) == 0:
                    # No more edges can be added - goal state
                    break

                max_action = None
                max_action_val = float("-inf")
                for e in legal_actions:
                    action_val = self.get_Q_val(e)

                    if action_val > max_action_val:
                        max_action = e
                        max_action_val = action_val

                self.make_move(max_action)

            # At goal state
            self.goal_state_update()

        return self.known_winners


    def load_model(self, checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint)