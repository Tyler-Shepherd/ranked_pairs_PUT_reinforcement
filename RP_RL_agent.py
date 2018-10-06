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
from networkx.algorithms.connectivity import local_edge_connectivity
from networkx.algorithms.connectivity import local_node_connectivity
import rpconfig
from collections import defaultdict
import matplotlib.pyplot as plt
from queue import PriorityQueue
import torch
from torch.autograd import Variable
import random
from pprint import pprint
import glob

# sys.path.append('..//..//node2vec-master//src')
# import node2vec
# import main as node2vecmain

import params as params

class RP_RL_stats():
    def __init__(self):
        self.num_nodes = 0
        self.stop_condition_hits = {1: 0, 2: 0, 3: 0}

        # loss over each profile
        self.running_loss = 0

# Util functions
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def edges2string(edges, I):
    m = len(I)
    gstring = list(str(0).zfill(m**2))
    for e in edges:
        gstring[(e[0] - min(I))*m + e[1] - min(I)] = '1'

    return ''.join(gstring)

# computes the plurality scores of candidates given an input profile
# input: profile of preferences as np matrix
# output: m-vector of plurality scores of candidates, normalized by n
def plurality_score(profile_matrix):
    (n,m) = np.shape(profile_matrix)
    pluralityscores = [0] * m
    for i in range(n):
        pluralityscores[profile_matrix[i,0]] += 1
    pluralityscores_normalized = list(1.*np.array(pluralityscores)/n)
    return pluralityscores_normalized

#computes the Borda scores of candidates given an input profile
# input: profile
# output: m-vector of Borda scores of candidates, normalized by n(m-1)
def borda_score(profile_matrix):
    (n,m) = np.shape(profile_matrix)
    bordascores = [0] * m
    for i in range(n):
        for j in range(m):
            bordascores[profile_matrix[i,j]] += (m - j)
    bordascores_normalized = list(1.*np.array(bordascores)/(n*(m-1)))
    return bordascores_normalized

#computes the Copeland scores of candidates
# input: wmg dict
# output: m-vector of Copeland scores of candidates, normalized by m-1 to [-1, 1]
def copeland_score(wmg):
    m = len(wmg.keys())
    copelandscores = [0] * m
    for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
        if wmg[cand1][cand2] > 0:
            copelandscores[cand1] += 1
            copelandscores[cand2] -= 1
    copelandscores_normalized = list(1.*np.array(copelandscores)/(m-1))
    return copelandscores_normalized

#computes the Maximin scores of candidates
# input: wmg dict
# output: m-vector of Maximin scores of candidates, normalized by n to [-1, 1]
def maximin_score(wmg):
    n = len(wmg.keys())
    maximinscores = [0] * n
    for cand in wmg.keys():
        maximinscores[cand] = min(i for (_, i) in wmg[cand].items())

    maximinscores_normalized = list(1.*np.array(maximinscores)/n)
    return maximinscores_normalized

# Vectorizes the wmg
# input: wmg
# output: vectorized weighted majority graph. sorted by candidates, then by opponents,
#   normalized by no. of voters
def vectorize_wmg(wmg):
    m = len(wmg)
    n = np.sum(np.abs([wmg[0][i] for i in range(1,m)]))
    wmg_vec = [wmg[i][j] for i in range(m) for j in range(m) if not j == i]
    wmg_vec_normalized = list(1.*np.array(wmg_vec)/n)
    return wmg_vec_normalized


# creates a positional matrix and vectorizes it
# input: profile
# intermediate: positional matrix posmat
#   posmat[i][j] = # voters ranking candidate i in position j
# output: vectorized positional matrix, sorted by candidate, then by position,
#   normalized by no. of voters
def profile2posmat(profile_matrix):
    (n,m) = np.shape(profile_matrix)
    posmat = np.zeros((m,m))

    for i in range(n):
        vote = profile_matrix[i, :]
        for pos in range(m):
            cand = vote[0, pos]
            posmat[cand][pos] += 1
    posmat_vec = posmat.flatten()
    posmat_vec_normalized = list(1.*np.array(posmat_vec)/n)
    return posmat_vec_normalized

# For node s, avg over all other nodes t of local edge connectivity = num edges needed to remove to disconnect s and t
def avg_edge_connectivity(G, I, s):
    total_connectivity = 0
    for t in I:
        if t != s:
            total_connectivity += local_edge_connectivity(G, s, t)

    avg_connectivity = total_connectivity / (len(I) - 1)
    # TODO: normalize
    return avg_connectivity

# For node s, avg over all other nodes t of local node connectivity = num nodes needed to remove to disconnect s and t
def avg_node_connectivity(G, I, s):
    total_connectivity = 0
    for t in I:
        if t != s:
            total_connectivity += local_node_connectivity(G, s, t)

    avg_connectivity = total_connectivity / (len(I) - 1)
    # TODO: normalize
    return avg_connectivity



class RP_RL_agent():
    def __init__(self, learning_rate = 0, loss_output_file = None):
        # Initialize learning model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(params.D_in, params.H1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(params.H1, params.H2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(params.H2, params.D_out)
        )

        # target fixed network to use in updating loss for stabilization
        # gets updated to self.model periodically
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(params.D_in, params.H1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(params.H1, params.H2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(params.H2, params.D_out)
        )

        # self.loss_fn = torch.nn.MSELoss(size_average=False)  # using mean squared error
        self.loss_fn = torch.nn.SmoothL1Loss(size_average=False) # Huber loss

        for p in self.target_model.parameters():
            p.requires_grad = False

        self.model.apply(init_weights)
        self.target_model.load_state_dict(self.model.state_dict())

        # loss reset every time it gets printed
        # can sum over multiple profiles
        self.running_loss = 0

        # total number of running nodes over all training
        self.running_nodes = 0

        self.loss_output_file = loss_output_file

        if params.optimizer_algo == 2:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.visited = {}

        self.stats = RP_RL_stats()

        # node2vec stuff
        # self.node2vec_args = node2vecmain.parse_args()
        # self.node2vec_args.directed = True
        # self.node2vec_args.weighted = False
        # self.node2vec_args.unweighted = True
        # self.node2vec_args.undirected = False
        # self.node2vec_args.output = "node2vec_output.emb"
        # self.node2vec_args.num_walks = 1
        # self.node2vec_args.walk_length = 1
        # self.node2vec_args.dimensions = 2

    '''
    Initializes environment for an iteration of learning
    env0 is a profile
    creates G_0 = initial state of RP graph (no edges)
    creates E_0 = initial state of unadded edges (all of them)
    resets visited set
    '''
    def initialize(self, env0):
        wmg = env0.getWmg()
        self.known_winners = set()
        self.I = list(wmg.keys())

        # save profile as matrix for use in features
        profile_matrix = []
        for p in env0.preferences:
            profile_matrix.append(p.getOrderVector())
        self.profile_matrix = np.asmatrix(profile_matrix)

        self.G_0 = nx.DiGraph()
        self.G_0.add_nodes_from(self.I)

        self.E_0 = nx.DiGraph()
        self.E_0.add_nodes_from(self.I)
        self.max_edge_weight = 0
        for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
            if wmg[cand1][cand2] > 0:
                self.E_0.add_edge(cand1, cand2, weight=wmg[cand1][cand2])
                self.max_edge_weight = max(self.max_edge_weight, wmg[cand1][cand2])

        self.E_0_really = self.E_0.copy()

        # Add any bridge edges from any tier in E_0 to G_0
        # These are guaranteed to never be in a cycle, so will always be in the final graph after RP procedure
        Gc = self.E_0.copy()
        scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if len(g.edges()) != 0]
        bridges = set(Gc.edges()) - set(itertools.chain(*scc))

        # Compute cycles
        # too slow for m50n50
        if params.use_cycles:
            cycles = nx.simple_cycles(Gc)

            self.num_cycles = 0
            self.edge_to_cycle_occurrence = {}

            for c in cycles:
                self.num_cycles += 1
                for i in range(len(c) - 1):
                    e0 = c[i]
                    e1 = c[i+1]
                    e = (e0, e1)

                    if e not in self.edge_to_cycle_occurrence:
                        self.edge_to_cycle_occurrence[e] = 0
                    self.edge_to_cycle_occurrence[e] += 1

                e = (c[-1], c[0])
                if e not in self.edge_to_cycle_occurrence:
                    self.edge_to_cycle_occurrence[e] = 0
                self.edge_to_cycle_occurrence[e] += 1

        self.G_0.add_edges_from(bridges)
        self.E_0.remove_edges_from(bridges)

        # enable for node2vec
        # node2vec_G = node2vec.Graph(self.E_0, True, self.node2vec_args.p, self.node2vec_args.q)
        # node2vec_G.preprocess_transition_probs()
        # walks = node2vec_G.simulate_walks(self.node2vec_args.num_walks, self.node2vec_args.walk_length)
        # self.node2vec_model = node2vecmain.learn_embeddings(walks, self.node2vec_args)

        self.visited = {}

        # compute voting rules scores
        self.plurality_scores = plurality_score(self.profile_matrix)
        self.borda_scores = borda_score(self.profile_matrix)
        self.copeland_scores = copeland_score(wmg)
        self.maximin_scores = maximin_score(wmg)

        self.vectorized_wmg = vectorize_wmg(wmg)
        self.posmat = profile2posmat(self.profile_matrix)
        self.adjacency_0 = nx.adjacency_matrix(self.E_0_really, nodelist=self.I).todense()

        if params.f_shape_reward:
            self.winners_visited = {}

        self.stats = RP_RL_stats()

    '''
    Resets G (the current RP graph), E (the graph of unadded edges) and K (the known winners)
    Does not reset the visited set
    '''
    def reset_environment(self):
        self.G = self.G_0.copy()
        self.E = self.E_0.copy()

        # Randomly initialize known winners
        self.K = set()

        for a in self.known_winners:
            if random.random() > 0.5:
                self.K.add(a)
        self.K = frozenset(self.K)

        if params.use_visited:
            self.update_visited()

    '''
    Returns -1 if not at goal state
    Returns 1 if E has no more edges
    Returns 2 if pruned
    Returns 3 if only one possible winner
    '''
    def at_goal_state(self, update_stats = 1):
        in_deg = self.G.in_degree(self.I)
        possible_winners = [x[0] for x in in_deg if x[1] == 0]

        # Stop Condition 2: Pruning. Possible winners are subset of known winners
        if set(possible_winners) <= self.K:
            self.stats.stop_condition_hits[2] += (1 * update_stats)
            if params.debug_mode >= 3:
                print("at_goal_state: 2")
            return 2, possible_winners

        # Stop Condition 1: E has no more edges
        if len(self.E.edges()) == 0:
            self.stats.stop_condition_hits[1] += (1 * update_stats)
            if params.debug_mode >= 3:
                print("at_goal_state: 1")
            return 1, possible_winners

        # Stop Condition 3: Exactly one node has indegree 0
        if len(possible_winners) == 1:
            self.stats.stop_condition_hits[3] += (1 * update_stats)
            if params.debug_mode >= 3:
                print("at_goal_state: 3")
            return 3, possible_winners

        return -1, possible_winners

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

    # Returns array of val to each power from 1 to i
    def polynomialize(self, val, i):
        return [val**j for j in range(1, i+1)]


    # Returns input layer features at current state taking action a
    # a is an edge
    def state_features(self, a):
        u = a[0]
        v = a[1]

        f = []

        # out/in degree
        if params.use_in_out:
            f.extend(self.polynomialize(self.safe_div(self.G.out_degree(u), self.E_0.out_degree(u)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(self.G.in_degree(u), self.E_0.in_degree(u)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(self.G.out_degree(v), self.E_0.out_degree(v)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(self.G.in_degree(v), self.E_0.in_degree(v)), params.num_polynomial))

        # total degree
        # f.extend(self.polynomialize(self.safe_div(self.G.out_degree(u) + self.G.in_degree(u), self.E_0.out_degree(u) + self.E_0.in_degree(u)), params.num_polynomial))
        # f.extend(self.polynomialize(self.safe_div(self.G.out_degree(v) + self.G.in_degree(v), self.E_0.out_degree(v) + self.E_0.in_degree(v)), params.num_polynomial))

        # binary "has out/in degree" features
        if params.use_in_out_binary:
            f.append(2 * int(self.G.out_degree(u) > 0) - 1)
            f.append(2 * int(self.G.in_degree(u) > 0) - 1)
            f.append(2 * int(self.G.out_degree(v) > 0) - 1)
            f.append(2 * int(self.G.in_degree(v) > 0) - 1)

        # known winners features
        if params.use_K:
            f.append(2 * int(u in self.K) - 1)
            f.append(2 * int(v in self.K) - 1)

        # voting rules scores
        if params.use_voting_rules:
            f.extend(self.polynomialize(self.plurality_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.plurality_scores[v], params.num_polynomial))
            f.extend(self.polynomialize(self.borda_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.borda_scores[v], params.num_polynomial))
            f.extend(self.polynomialize(self.copeland_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.copeland_scores[v], params.num_polynomial))
            f.extend(self.polynomialize(self.maximin_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.maximin_scores[v], params.num_polynomial))

        if params.use_vectorized_wmg:
            f.extend(self.vectorized_wmg)

        if params.use_posmat:
            f.extend(self.posmat)

        # num cycles feature
        if params.use_cycles:
            if a in self.edge_to_cycle_occurrence:
                f.extend(self.polynomialize(self.safe_div(self.edge_to_cycle_occurrence[a], self.num_cycles), params.num_polynomial))
            else:
                f.extend(self.polynomialize(0, params.num_polynomial))

        # visited feature
        if params.use_visited:
            f.extend(self.polynomialize(self.get_num_times_visited(), params.num_polynomial))

        # edge weight
        if params.use_edge_weight:
            f.extend(self.polynomialize(self.E_0_really[u][v]['weight'] / self.max_edge_weight, params.num_polynomial))

        # adjacency matrix if a is added
        if params.use_adjacency_matrix:
            self.G.add_edge(u,v)
            adjacency = nx.adjacency_matrix(self.G, nodelist = self.I).todense()
            adjacency = np.multiply(adjacency, self.adjacency_0)
            adjacency_normalized = np.divide(adjacency, params.n)
            f.extend(adjacency_normalized.flatten().tolist()[0])
            self.G.remove_edge(u,v)

        # K representation
        if params.use_K_representation:
            K_list = []
            for i in self.I:
                if i in self.K:
                    K_list.append(1)
                else:
                    K_list.append(0)
            f.extend(K_list)

        # tier adjacency matrix
        if params.use_tier_adjacency_matrix:
            legal_actions = self.get_legal_actions()
            legal_actions.remove(a)
            T = np.zeros((int(params.m), int(params.m)))
            for (c1,c2) in legal_actions:
                T[c1,c2] = 1
            T_vec = list(T.flatten())
            f.extend(T_vec)

        # edge and node connectivity
        if params.use_connectivity:
            f.extend(self.polynomialize(avg_edge_connectivity(self.G, self.I, u), params.num_polynomial))
            f.extend(self.polynomialize(avg_edge_connectivity(self.G, self.I, v), params.num_polynomial))
            f.extend(self.polynomialize(avg_node_connectivity(self.G, self.I, u), params.num_polynomial))
            f.extend(self.polynomialize(avg_node_connectivity(self.G, self.I, v), params.num_polynomial))

        # node2vec every time
        # G_with_weights = nx.DiGraph()
        # G_with_weights.add_nodes_from(self.I)
        # for (cand1, cand2) in self.G.edges():
        #     G_with_weights.add_edge(cand1, cand2, weight=self.E_0_really[cand1][cand2]['weight'])
        # node2vec_G = node2vec.Graph(G_with_weights, True, self.node2vec_args.p, self.node2vec_args.q)
        # node2vec_G.preprocess_transition_probs()
        # walks = node2vec_G.simulate_walks(self.node2vec_args.num_walks, self.node2vec_args.walk_length)
        # self.node2vec_model = node2vecmain.learn_embeddings(walks, self.node2vec_args)

        # node2vec features
        # node2vec_u = self.node2vec_model.wv[str(u)]
        # node2vec_v = self.node2vec_model.wv[str(v)]
        # node2vec_uv = np.append(node2vec_u, node2vec_v)

        # node2vec_f = np.append(node2vec_uv, np.array(f))

        if params.debug_mode >= 3:
            print("features", f)

        return Variable(torch.from_numpy(np.array(f)).float())
        # return Variable(torch.from_numpy(node2vec_f).float())

    def get_Q_val(self, a, use_target_net=False):
        state_features = self.state_features(a)

        if use_target_net:
            return self.target_model(state_features)
        return self.model(state_features)

    # Adds new PUT-winners to self.known_winners
    def goal_state_update(self):
        G_in_degree = self.G.in_degree(self.I)
        to_be_added = set([x[0] for x in G_in_degree if x[1] == 0])

        new_winners = []

        for c in to_be_added:
            if c not in self.known_winners:
                self.known_winners.add(c)
                new_winners.append(c)

        if params.debug_mode >= 2:
            print('goal state with new winners', new_winners)

        return new_winners

    def print_model(self, output_file, f_print_to_console=True):
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
            if f_print_to_console:
                print(p)
        if output_file != "":
            output_file.write('\n')
            output_file.flush()

    '''
    Adds edge a from E to G
    And performs reductions
    '''
    def make_move(self, a, f_testing = False):
        if params.debug_mode >= 2:
            print('making move', a)

        self.G.add_edges_from([a])
        self.E.remove_edges_from([a])

        # Remove inconsistent edges from E
        G_transitive_closure = nx.transitive_closure(self.G)
        Ec = self.E.copy().edges()
        for e in Ec:
            if G_transitive_closure.has_edge(e[1], e[0]):
                self.E.remove_edges_from([e])

        # Add bridge edges from E to G
        T = self.get_legal_actions()

        Gc = self.G.copy()
        Gc.add_edges_from(T)
        scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if len(g.edges()) != 0]
        bridges = set(Gc.edges()) - set(itertools.chain(*scc))
        self.G.add_edges_from(bridges)
        self.E.remove_edges_from(bridges)
        T = list(set(T) - bridges)

        # Remove "redundant edges": if there is already path from e[0] to e[1], can immediately add e
        redundant_edges = set()
        for e in T:
            if G_transitive_closure.has_edge(e[0], e[1]):
                redundant_edges.add(e)
                self.G.add_edges_from([e])
                self.E.remove_edges_from([e])

        self.stats.num_nodes += 1

        if params.use_visited:
            self.update_visited()

        if not f_testing:
            self.running_nodes += 1

            if self.running_nodes % params.print_loss_every == 0:
                print("*******LOSS:", self.running_loss / params.print_loss_every)
                if self.loss_output_file:
                    self.loss_output_file.write(str(self.running_nodes) + '\t' + str(self.running_loss / params.print_loss_every) + '\n')
                    self.loss_output_file.flush()

                self.running_loss = 0



    def reward(self):
        current_state, possible_winners = self.at_goal_state(update_stats=0)

        if current_state == -1:
            # Not a goal state
            reward_val =  0
        elif current_state == 2:
            # Pruning state
            reward_val = -1
        else:
            # Found a new winner
            if params.f_shape_reward:
                reward_val = 0
                for w in possible_winners:
                    if w not in self.K:
                        if w not in self.winners_visited:
                            self.winners_visited[w] = 0
                        self.winners_visited[w] += 1
                        reward_val += 1.0 / self.winners_visited[w]
            else:
                reward_val = 1

        return torch.tensor(reward_val, dtype = torch.float32)

    '''
    Updates model based on chosen optimizer
    Note: Does not affect or update state at all
    '''
    def update_q(self, learning_rate, old_q_value, new_q_value):

        new_q_value = Variable(new_q_value)

        # Compute loss
        loss = self.loss_fn(old_q_value, new_q_value)

        self.stats.running_loss += loss.item()
        self.running_loss += loss.item()

        if params.debug_mode >= 2:
            print("loss, old_q_val, new_q_val", loss.item(), old_q_value.item(), new_q_value.item())

        if params.optimizer_algo == 1:
            # Gradient descent

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
        elif params.optimizer_algo == 2:
            # Adam

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()


    def save_model(self, id):
        # Save the model
        torch.save(self.model.state_dict(), "results_RP_RL_main" + str(id) + "_model.pth.tar")
        print("model saved")


    def test_model(self, test_env):
        self.initialize(test_env)

        times_discovered = []
        num_iters_to_find_all_winners = 0

        # Sample using model greedily
        # Test with fixed number of iterations
        with torch.no_grad():
            for iter in range(params.num_test_iterations):
                self.reset_environment()
                self.K = frozenset(self.known_winners)

                num_iters_to_find_all_winners += 1

                while self.at_goal_state()[0] == -1:
                    legal_actions = self.get_legal_actions()

                    if len(legal_actions) == 0:
                        # No more edges can be added - goal state
                        break

                    # for random action selection testing
                    # max_action = legal_actions[random.randint(0, len(legal_actions) - 1)]

                    max_action = None
                    max_action_val = float("-inf")
                    for e in legal_actions:
                        action_val = self.get_Q_val(e)

                        if action_val > max_action_val:
                            max_action = e
                            max_action_val = action_val

                    assert (max_action is not None)

                    self.make_move(max_action, f_testing = True)

                # At goal state
                new_winners = self.goal_state_update()

                for c in new_winners:
                    times_discovered.append(iter)

        return self.known_winners, times_discovered, num_iters_to_find_all_winners


    # tests "how many samples to find all winners"
    def test_model_v2(self, test_env, true_winners):
        self.initialize(test_env)

        times_discovered = {}
        num_iters_to_find_all_winners = 0

        # Sample using model greedily
        with torch.no_grad():
            # Test till found all winners
            while self.known_winners != true_winners and num_iters_to_find_all_winners <= params.cutoff_testing_iterations:
                if not self.known_winners < true_winners:
                    print(self.known_winners, true_winners)
                assert self.known_winners < true_winners

                self.reset_environment()
                self.K = frozenset(self.known_winners)

                num_iters_to_find_all_winners += 1

                while self.at_goal_state()[0] == -1:
                    legal_actions = self.get_legal_actions()

                    if len(legal_actions) == 0:
                        # No more edges can be added - goal state
                        break

                    # for random action selection testing
                    # selected_action = legal_actions[random.randint(0, len(legal_actions) - 1)]

                    # Boltzmann
                    q_vals = []
                    q_vals_blah = []
                    for e in legal_actions:
                        q_vals_blah.append(self.get_Q_val(e).item())
                        q_vals.append(exp(self.get_Q_val(e).item() / params.tau_for_testing))
                    q_sum = sum(q_vals)
                    probs = []
                    for v in q_vals:
                        probs.append(v / q_sum)
                    legal_actions_index = [i for i in range(len(legal_actions))]
                    selected_action = legal_actions[np.random.choice(legal_actions_index, p=probs)]

                    assert selected_action is not None

                    self.make_move(selected_action, f_testing = True)

                # At goal state
                new_winners = self.goal_state_update()

                for c in new_winners:
                    times_discovered[c] = num_iters_to_find_all_winners

        return self.known_winners, times_discovered, num_iters_to_find_all_winners


    def load_model(self, checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint)
        self.target_model.load_state_dict(self.model.state_dict())
        print("loaded model")

    def get_current_state(self):
        return [self.G.copy(), self.E.copy(), self.K.copy()]

    def set_state(self, new_state):
        # new_state is of form [G, E, K]
        self.G = new_state[0]
        self.E = new_state[1]
        self.K = new_state[2]

    def safe_div(self, num, denom):
        if denom == 0:
            return 0
        return num / denom

    def state_as_string(self):
        state_str = edges2string(self.G.edges(), self.I) + edges2string(self.E.edges(), self.I)
        K_str = ""
        for i in self.I:
            if i in self.K:
                K_str += "1"
            else:
                K_str += "0"
        state_str += K_str
        return state_str

    def update_visited(self):
        state_str = self.state_as_string()
        if state_str in self.visited:
            self.visited[state_str] += 1
        else:
            self.visited[state_str] = 1

    def get_num_times_visited(self):
        state_str = self.state_as_string()
        if state_str in self.visited:
            return self.visited[state_str]
        return 0

