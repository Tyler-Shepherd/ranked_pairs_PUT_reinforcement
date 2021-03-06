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
import RP_utils

class RP_RL_stats():
    def __init__(self):
        self.num_nodes = 0
        self.stop_condition_hits = {1: 0, 2: 0, 3: 0}

        # loss over each profile, reset every profile
        self.running_loss = 0

        self.time_for_connectivity = 0

        self.num_iters_reset_skipped = 0

        self.num_hashed = 0


class RP_RL_agent():
    def __init__(self, model, learning_rate = 0, loss_output_file = None):
        # Initialize learning model
        self.model = model

        # target fixed network to use in updating loss for stabilization
        # gets updated to self.model periodically
        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        self.target_model = copy.deepcopy(model)

        # self.loss_fn = torch.nn.MSELoss(size_average=False)  # using mean squared error
        self.loss_fn = torch.nn.SmoothL1Loss(size_average=False) # Huber loss

        for p in self.target_model.parameters():
            p.requires_grad = False

        self.target_model.load_state_dict(self.model.state_dict())

        # loss reset every time it gets printed
        # can sum over multiple profiles
        self.running_loss = 0

        # total number of running nodes over all training
        self.running_nodes = 0

        self.loss_output_file = loss_output_file

        if params.optimizer_algo == 2:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif params.optimizer_algo == 3:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.visited = {}

        self.stats = RP_RL_stats()

        # node2vec stuff (unused since node2vec takes an unreasonably long time)
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
        if params.use_voting_rules or params.use_posmat:
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
        if params.use_voting_rules:
            self.plurality_scores = RP_utils.plurality_score(self.profile_matrix)
            self.borda_scores = RP_utils.borda_score(self.profile_matrix)
            self.copeland_scores = RP_utils.copeland_score(wmg)
            self.maximin_scores = RP_utils.maximin_score(wmg)

        if params.use_vectorized_wmg:
            self.vectorized_wmg = RP_utils.vectorize_wmg(wmg)

        if params.use_posmat:
            self.posmat = RP_utils.profile2posmat(self.profile_matrix)

        if params.use_adjacency_matrix:
            self.adjacency_0 = nx.adjacency_matrix(self.E_0_really, nodelist=self.I).todense()

        if params.f_shape_reward:
            self.winners_visited = {}

        if params.use_betweenness_centrality:
            # should be O(n^3)
            # only takes 0.016 seconds on m50
            # http://www.algo.uni-konstanz.de/publications/b-fabc-01.pdf
            self.betweenness_centralities = nx.betweenness_centrality(self.E_0_really, normalized=True)

        self.stats = RP_RL_stats()

    '''
    Resets G (the current RP graph), E (the graph of unadded edges) and K (the known winners)
    Does not reset the visited set
    Randomly initializes K from known_winners - if iter_to_find_winner supplied, uses that to determine probability of including each winner
    If winners_distribution provided, uses that to initialize K (winners_distribution is more accurate than iter_to_find_winner)
    '''
    def reset_environment(self, iter_to_find_winner = None, winners_distribution = None):
        self.G = self.G_0.copy()
        self.E = self.E_0.copy()

        # Randomly initialize known winners
        self.K = set()

        if winners_distribution is not None:
            total_iters = 1000
            true_winners = winners_distribution.keys()
            probs = []
            for a in true_winners:
                prob_of_adding = winners_distribution[a] / (total_iters + 1)
                probs.append(prob_of_adding)
                if random.random() < prob_of_adding:
                    self.K.add(a)
            if true_winners == self.K:
                # pointless to train if K is already the true winners set
                self.stats.num_iters_reset_skipped += 1
                return -1
        elif iter_to_find_winner is not None:
            max_num_iters = max(i for i in iter_to_find_winner.values())
            for a in self.known_winners:
                if random.random() > (1 - iter_to_find_winner[a] / (max_num_iters + 1)):
                    self.K.add(a)
        else:
            for a in self.known_winners:
                if random.random() > 0.5:
                    self.K.add(a)
        self.K = frozenset(self.K)

        if params.use_visited:
            self.update_visited()

        return 0

    '''
    Returns -1 if not at goal state
    Returns 1 if E has no more edges
    Returns 2 if pruned
    Returns 3 if only one possible winner
    Returns the possible winners (cands with indegree 0) as second value
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


    '''
    Returns input layer features at current state taking action a
    a is an edge
    '''
    def state_features(self, a):
        u = a[0]
        v = a[1]

        f = []

        # out/in degree
        if params.use_in_out:
            f.extend(RP_utils.polynomialize(self.G.out_degree(u) / params.m, params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.G.in_degree(u) / params.m, params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.G.out_degree(v) / params.m, params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.G.in_degree(v) / params.m, params.num_polynomial))

        if params.use_in_out_relative:
            f.extend(RP_utils.polynomialize(RP_utils.safe_div(self.G.out_degree(u), self.E_0.out_degree(u)), params.num_polynomial))
            f.extend(RP_utils.polynomialize(RP_utils.safe_div(self.G.in_degree(u), self.E_0.in_degree(u)), params.num_polynomial))
            f.extend(RP_utils.polynomialize(RP_utils.safe_div(self.G.out_degree(v), self.E_0.out_degree(v)), params.num_polynomial))
            f.extend(RP_utils.polynomialize(RP_utils.safe_div(self.G.in_degree(v), self.E_0.in_degree(v)), params.num_polynomial))

        # total degree
        if params.use_total_degree:
            f.extend(RP_utils.polynomialize(RP_utils.safe_div(self.G.out_degree(u) + self.G.in_degree(u), self.E_0.out_degree(u) + self.E_0.in_degree(u)), params.num_polynomial))
            f.extend(RP_utils.polynomialize(RP_utils.safe_div(self.G.out_degree(v) + self.G.in_degree(v), self.E_0.out_degree(v) + self.E_0.in_degree(v)), params.num_polynomial))

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

        if params.use_K_big:
            f.append(int(u in self.K) * 100)
            f.append(int(v in self.K) * 100)

        # voting rules scores
        if params.use_voting_rules:
            f.extend(RP_utils.polynomialize(self.plurality_scores[u], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.plurality_scores[v], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.borda_scores[u], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.borda_scores[v], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.copeland_scores[u], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.copeland_scores[v], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.maximin_scores[u], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.maximin_scores[v], params.num_polynomial))

        if params.use_vectorized_wmg:
            f.extend(self.vectorized_wmg)

        if params.use_posmat:
            f.extend(self.posmat)

        # num cycles feature
        if params.use_cycles:
            if a in self.edge_to_cycle_occurrence:
                f.extend(RP_utils.polynomialize(RP_utils.safe_div(self.edge_to_cycle_occurrence[a], self.num_cycles), params.num_polynomial))
            else:
                f.extend(RP_utils.polynomialize(0, params.num_polynomial))

        # visited feature
        if params.use_visited:
            f.extend(RP_utils.polynomialize(self.get_num_times_visited(), params.num_polynomial))

        # edge weight
        if params.use_edge_weight:
            f.extend(RP_utils.polynomialize(self.E_0_really[u][v]['weight'] / self.max_edge_weight, params.num_polynomial))

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
            start_connect = time.perf_counter()
            f.extend(RP_utils.polynomialize(RP_utils.avg_edge_connectivity(self.G, self.I, u), params.num_polynomial))
            f.extend(RP_utils.polynomialize(RP_utils.avg_edge_connectivity(self.G, self.I, v), params.num_polynomial))
            f.extend(RP_utils.polynomialize(RP_utils.avg_node_connectivity(self.G, self.I, u), params.num_polynomial))
            f.extend(RP_utils.polynomialize(RP_utils.avg_node_connectivity(self.G, self.I, v), params.num_polynomial))
            self.stats.time_for_connectivity += (time.perf_counter() - start_connect)

        if params.use_betweenness_centrality:
            f.extend(RP_utils.polynomialize(self.betweenness_centralities[u], params.num_polynomial))
            f.extend(RP_utils.polynomialize(self.betweenness_centralities[v], params.num_polynomial))

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

    '''
    Gets q value of current state if action a taken
    uses frozen target network if use_target_net
    '''
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
                        winner_reward_val = 1 - self.winners_visited[w] / (params.num_training_iterations / 20 + self.winners_visited[w])
                        reward_val = max(reward_val, winner_reward_val)
                        self.winners_visited[w] += 1
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
        else:
            # Adam or any other optimizer

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


    '''
    Tests num of PUT_winners that can be found in params.num_test_iterations samples
    test_env is profile to test on
    begin_time is the time.perf_counter this was called at
    val_testing is whether or not this test is being done for validation
    '''
    def test_model(self, test_env, begin_time = 0, val_testing = 0):
        self.initialize(test_env)

        times_discovered = {}
        runtimes_discovered = {}
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
                    if params.test_with_random:
                        max_action = legal_actions[random.randint(0, len(legal_actions) - 1)]

                    # pure greedy
                    # max_action = None
                    # max_action_val = float("-inf")
                    # for e in legal_actions:
                    #     action_val = self.get_Q_val(e)
                    #
                    #     if action_val > max_action_val:
                    #         max_action = e
                    #         max_action_val = action_val
                    #

                    # Boltzmann q
                    elif not params.test_with_LP:
                        q_vals = []
                        for e in legal_actions:
                            q_vals.append(exp(self.get_Q_val(e).item() / params.tau_for_testing))
                        q_sum = sum(q_vals)
                        probs = []
                        for v in q_vals:
                            probs.append(v / q_sum)
                        legal_actions_index = [i for i in range(len(legal_actions))]
                        max_action = legal_actions[np.random.choice(legal_actions_index, p=probs)]

                    # Boltzmann LPwinners
                    else:
                        priorities = []
                        for e in legal_actions:
                            Gc = self.G.copy()
                            Gc.add_edges_from([e])
                            G_in_degree = Gc.in_degree(self.I)
                            potential_winners = set([x[0] for x in G_in_degree if x[1] == 0])
                            priority = len(potential_winners - self.known_winners)
                            priorities.append(exp(priority / params.tau_for_testing))
                        q_sum = sum(priorities)
                        probs = []
                        for v in priorities:
                            probs.append(v / q_sum)
                        legal_actions_index = [i for i in range(len(legal_actions))]
                        max_action = legal_actions[np.random.choice(legal_actions_index, p=probs)]

                    assert (max_action is not None)

                    self.make_move(max_action, f_testing = True)

                # At goal state
                new_winners = self.goal_state_update()

                for c in new_winners:
                    runtimes_discovered[c] = time.perf_counter() - begin_time
                    times_discovered[c] = iter

        return self.known_winners, times_discovered, num_iters_to_find_all_winners, runtimes_discovered


    '''
    Tests "how many samples to find all winners"
    test_env is profile
    true_winners is the set of the actual winners for test_env
    begin_time is time.perf_counter this was called at
    '''
    def test_model_v2(self, test_env, true_winners, begin_time = 0):
        self.initialize(test_env)

        times_discovered = {}
        runtimes_discovered = {}
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
                    if params.test_with_random:
                        selected_action = legal_actions[random.randint(0, len(legal_actions) - 1)]

                    # LPwinners (w/ Boltzmann)
                    elif params.test_with_LP:
                        priorities = []
                        for e in legal_actions:
                            Gc = self.G.copy()
                            Gc.add_edges_from([e])
                            G_in_degree = Gc.in_degree(self.I)
                            potential_winners = set([x[0] for x in G_in_degree if x[1] == 0])
                            priority = len(potential_winners - self.known_winners)
                            priorities.append(exp(priority / params.tau_for_testing))
                        q_sum = sum(priorities)
                        probs = []
                        for v in priorities:
                            probs.append(v / q_sum)
                        legal_actions_index = [i for i in range(len(legal_actions))]
                        selected_action = legal_actions[np.random.choice(legal_actions_index, p=probs)]

                    # Boltzmann
                    else:
                        q_vals = []
                        for e in legal_actions:
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
                    runtimes_discovered[c] = time.perf_counter() - begin_time
                    times_discovered[c] = num_iters_to_find_all_winners

        return self.known_winners, times_discovered, num_iters_to_find_all_winners, runtimes_discovered

    def get_current_state(self):
        other_vars = []
        other_vars.append(self.E_0.copy())
        if params.use_voting_rules:
            other_vars.append(copy.deepcopy(self.plurality_scores))
            other_vars.append(copy.deepcopy(self.borda_scores))
            other_vars.append(copy.deepcopy(self.copeland_scores))
            other_vars.append(copy.deepcopy(self.maximin_scores))
        else:
            other_vars.append(None)
            other_vars.append(None)
            other_vars.append(None)
            other_vars.append(None)

        if params.use_vectorized_wmg:
            other_vars.append(copy.deepcopy(self.vectorized_wmg))
        else:
            other_vars.append(None)

        if params.use_posmat:
            other_vars.append(copy.deepcopy(self.posmat))
        else:
            other_vars.append(None)

        if params.use_adjacency_matrix:
            other_vars.append(copy.deepcopy(self.adjacency_0))
        else:
            other_vars.append(None)

        other_vars.append(self.E_0_really.copy())
        other_vars.append(copy.deepcopy(self.max_edge_weight))

        return [self.G.copy(), self.E.copy(), self.K.copy(), other_vars]

    def set_state(self, new_state):
        # new_state is of form [G, E, K, other_vars] (same as get_state)
        self.G = new_state[0]
        self.E = new_state[1]
        self.K = new_state[2]
        other_vars = new_state[3]
        self.E_0 = other_vars[0]
        self.plurality_scores = other_vars[1]
        self.borda_scores = other_vars[2]
        self.copeland_scores = other_vars[3]
        self.maximin_scores = other_vars[4]
        self.vectorized_wmg = other_vars[5]
        self.posmat = other_vars[6]
        self.adjacency_0 = other_vars[7]
        self.E_0_really = other_vars[8]
        self.max_edge_weight = other_vars[9]

    def state_as_string(self):
        state_str = RP_utils.edges2string(self.G.edges(), self.I) + RP_utils.edges2string(self.E.edges(), self.I)
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

