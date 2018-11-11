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

        # loss over each profile
        self.running_loss = 0

        self.time_for_connectivity = 0

        self.num_iters_reset_skipped = 0

        self.num_hashed = 0


class RP_RL_agent_PUT():
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

        self.stats = RP_RL_stats()

    '''
    Initializes environment for an iteration of learning
    env0 is a profile
    creates G_0 = initial state of RP graph (no edges)
    creates E_0 = initial state of unadded edges (all of them)
    adds initial node to frontier
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

        self.visited = set()

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

        # frontier contains tuples of (e, G, E, depth)
        # e is edge that when added produced (G, E)
        self.frontier = []
        first_T = self.get_tier(self.G_0, self.E_0, 0)
        self.frontier.extend(first_T)

    '''
    Resets to start of PUT search
    Resets frontier
    Resets visited set
    '''
    def reset_environment(self, iter_to_find_winner = None, winners_distribution = None):
        self.frontier = []
        first_T = self.get_tier(self.G_0, self.E_0, 0)
        self.frontier.extend(first_T)

        self.visited = set()

        self.known_winners = set()

        return 0

    '''
    Returns -1 if nodes still in frontier
    Return 1 if frontier empty (fully completed search)
    '''
    def at_goal_state(self, update_stats = 1):
        if len(self.frontier) == 0:
            return 1, None
        else:
            return -1, None

    '''
    Returns -1 if (G,E) not a terminal state
    Returns 2 if terminal and found new winner
    Returns 1 if terminal and not found new winner
    Second return val is new winners if exist
    '''
    def at_terminal_state(self, G, E, update_stats=1):
        in_deg = G.in_degree(self.I)
        possible_winners = [x[0] for x in in_deg if x[1] == 0]

        # Stop Condition 2: Pruning. Possible winners are subset of known winners
        if set(possible_winners) <= self.known_winners:
            self.stats.stop_condition_hits[2] += (1 * update_stats)
            if params.debug_mode >= 3:
                print("at_goal_state: pruned")
            return 1, None

        # Stop Condition 1: E has no more edges
        if len(E.edges()) == 0:
            self.stats.stop_condition_hits[1] += (1 * update_stats)
            if params.debug_mode >= 3:
                print("at_goal_state: 1")
            new_winners = set(possible_winners) - self.known_winners
            if len(new_winners) > 0:
                return 2, new_winners
            return 1, None

        # Stop Condition 3: Exactly one node has indegree 0
        if len(possible_winners) == 1:
            self.stats.stop_condition_hits[3] += (1 * update_stats)
            if params.debug_mode >= 3:
                print("at_goal_state: 3")
            return 2, possible_winners

        return -1, None


    '''
    returns list of (e, G, E, depth) that are next tier
    '''
    def get_tier(self, G, E, depth):
        if len(E.edges()) == 0:
            # There are no possible actions
            return []

        G_transitive_closure = nx.transitive_closure(G)

        max_weight = max([(d['weight']) for (u, v, d) in E.edges(data=True)])
        T = [(u, v) for (u, v, d) in E.edges(data=True) if d['weight'] == max_weight]

        tier = []
        for e in T:
            if not G_transitive_closure.has_edge(e[1], e[0]):
                Gc = G.copy()
                Ec = E.copy()
                Gc.add_edges_from([e])
                Ec.remove_edges_from([e])
                tier.append((e, Gc, Ec, depth+1))

        return tier

    '''
    Returns frontier
    If len(frontier) > 1000, just returns 1000 randomly
    '''
    def get_legal_actions(self):
        max_len = 1000
        if len(self.frontier) > max_len:
            choices = np.random.choice(len(self.frontier), max_len, replace=False)
            return [self.frontier[i] for i in choices]
        return self.frontier

    '''
    Returns input layer features at current state taking action a
    a is tuple (edge, G, E, depth)
    '''
    def state_features(self, a):
        e = a[0]
        G = a[1]
        E = a[2]
        depth = a[3]

        u = e[0]
        v = e[1]

        f = []

        # out/in degree
        if params.use_in_out:
            f.extend(RP_utils.polynomialize(G.out_degree(u) / params.m, params.num_polynomial))
            f.extend(RP_utils.polynomialize(G.in_degree(u) / params.m, params.num_polynomial))
            f.extend(RP_utils.polynomialize(G.out_degree(v) / params.m, params.num_polynomial))
            f.extend(RP_utils.polynomialize(G.in_degree(v) / params.m, params.num_polynomial))

        if params.use_in_out_relative:
            assert 1==2
            f.extend(self.polynomialize(self.safe_div(self.G.out_degree(u), self.E_0.out_degree(u)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(self.G.in_degree(u), self.E_0.in_degree(u)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(self.G.out_degree(v), self.E_0.out_degree(v)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(self.G.in_degree(v), self.E_0.in_degree(v)), params.num_polynomial))

        # total degree
        if params.use_total_degree:
            assert 1 == 2
            f.extend(self.polynomialize(self.safe_div(self.G.out_degree(u) + self.G.in_degree(u), self.E_0.out_degree(u) + self.E_0.in_degree(u)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(self.G.out_degree(v) + self.G.in_degree(v), self.E_0.out_degree(v) + self.E_0.in_degree(v)), params.num_polynomial))

        # binary "has out/in degree" features
        if params.use_in_out_binary:
            assert 1 == 2
            f.append(2 * int(self.G.out_degree(u) > 0) - 1)
            f.append(2 * int(self.G.in_degree(u) > 0) - 1)
            f.append(2 * int(self.G.out_degree(v) > 0) - 1)
            f.append(2 * int(self.G.in_degree(v) > 0) - 1)

        # known winners features
        if params.use_K:
            f.append(2 * int(u in self.known_winners) - 1)
            f.append(2 * int(v in self.known_winners) - 1)

        if params.use_K_big:
            assert 1==2
            f.append(int(u in self.K) * 100)
            f.append(int(v in self.K) * 100)

        # voting rules scores
        if params.use_voting_rules:
            assert 1 == 2
            f.extend(self.polynomialize(self.plurality_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.plurality_scores[v], params.num_polynomial))
            f.extend(self.polynomialize(self.borda_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.borda_scores[v], params.num_polynomial))
            f.extend(self.polynomialize(self.copeland_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.copeland_scores[v], params.num_polynomial))
            f.extend(self.polynomialize(self.maximin_scores[u], params.num_polynomial))
            f.extend(self.polynomialize(self.maximin_scores[v], params.num_polynomial))

        if params.use_vectorized_wmg:
            assert 1 == 2
            f.extend(self.vectorized_wmg)

        if params.use_posmat:
            assert 1 == 2
            f.extend(self.posmat)

        # num cycles feature
        if params.use_cycles:
            assert 1 == 2
            if a in self.edge_to_cycle_occurrence:
                f.extend(self.polynomialize(self.safe_div(self.edge_to_cycle_occurrence[a], self.num_cycles), params.num_polynomial))
            else:
                f.extend(self.polynomialize(0, params.num_polynomial))

        # visited feature
        if params.use_visited:
            assert 1 == 2
            f.extend(self.polynomialize(self.get_num_times_visited(), params.num_polynomial))

        # edge weight
        if params.use_edge_weight:
            assert 1 == 2
            f.extend(self.polynomialize(self.E_0_really[u][v]['weight'] / self.max_edge_weight, params.num_polynomial))

        # adjacency matrix if a is added
        if params.use_adjacency_matrix:
            assert 1 == 2
            self.G.add_edge(u,v)
            adjacency = nx.adjacency_matrix(self.G, nodelist = self.I).todense()
            adjacency = np.multiply(adjacency, self.adjacency_0)
            adjacency_normalized = np.divide(adjacency, params.n)
            f.extend(adjacency_normalized.flatten().tolist()[0])
            self.G.remove_edge(u,v)

        # K representation
        if params.use_K_representation:
            assert 1 == 2
            K_list = []
            for i in self.I:
                if i in self.K:
                    K_list.append(1)
                else:
                    K_list.append(0)
            f.extend(K_list)

        # tier adjacency matrix
        if params.use_tier_adjacency_matrix:
            assert 1 == 2
            legal_actions = self.get_legal_actions()
            legal_actions.remove(a)
            T = np.zeros((int(params.m), int(params.m)))
            for (c1,c2) in legal_actions:
                T[c1,c2] = 1
            T_vec = list(T.flatten())
            f.extend(T_vec)

        # edge and node connectivity
        if params.use_connectivity:
            assert 1 == 2
            start_connect = time.perf_counter()
            f.extend(self.polynomialize(RP_utils.avg_edge_connectivity(self.G, self.I, u), params.num_polynomial))
            f.extend(self.polynomialize(RP_utils.avg_edge_connectivity(self.G, self.I, v), params.num_polynomial))
            f.extend(self.polynomialize(RP_utils.avg_node_connectivity(self.G, self.I, u), params.num_polynomial))
            f.extend(self.polynomialize(RP_utils.avg_node_connectivity(self.G, self.I, v), params.num_polynomial))
            self.stats.time_for_connectivity += (time.perf_counter() - start_connect)

        if params.use_betweenness_centrality:
            assert 1 == 2
            f.extend(self.polynomialize(self.betweenness_centralities[u], params.num_polynomial))
            f.extend(self.polynomialize(self.betweenness_centralities[v], params.num_polynomial))

        f.append(depth)

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

    '''
    Do nothing
    '''
    def goal_state_update(self):
        return

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
    Returns 1 if (G, E) previous visited
    Else returns 0 and adds to cache to add_to_cache
    '''
    def in_cache(self, G, E, add_to_cache):
        # Check cache
        hash_state = hash(str(G.edges()) + str(E.edges()))
        if hash_state in self.visited:
            self.stats.num_hashed += 1
            if params.debug_mode == 3:
                print("hashed in outer hashtable")
            return 1
        if add_to_cache:
            self.visited.add(hash_state)
        return 0

    '''
    a is tuple (edge, G, E, depth)
    G already has e added and E already has e removed
    Performs reductions
    Adds all next nodes to frontier (removing those that are cached)
    Saves reward as self var
    '''
    def make_move(self, a, f_testing = False):
        if params.debug_mode >= 2:
            print('making move', a)

        self.frontier.remove(a)

        e = a[0]
        G = a[1]
        E = a[2]
        depth = a[3]

        T = self.get_tier(G, E, depth)
        T_edges = [e_T[0] for e_T in T]

        # check cache
        # if self.in_cache(G, E, add_to_cache = True):
        #     self.current_reward = -depth
        #     return

        for e_next, G_next, E_next, depth_next in T:
            # Remove inconsistent edges
            G_transitive_closure = nx.transitive_closure(G_next)
            Ec = E_next.copy().edges()
            for e in Ec:
                if G_transitive_closure.has_edge(e[1], e[0]):
                    E_next.remove_edges_from([e])

            T_next = T_edges.copy()
            T_next.remove(e_next)

            # Add bridge edges
            Gc = G_next.copy()
            Gc.add_edges_from(T_next)
            scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if len(g.edges()) != 0]
            bridges = set(Gc.edges()) - set(itertools.chain(*scc))
            G_next.add_edges_from(bridges)
            E_next.remove_edges_from(bridges)
            T_next = list(set(T_next) - bridges)

            # Remove "redundant edges": if there is already path from e[0] to e[1], can immediately add e
            for e_T in T_next:
                if G_transitive_closure.has_edge(e_T[0], e_T[1]):
                    G_next.add_edges_from([e_T])
                    E_next.remove_edges_from([e_T])

            if self.in_cache(G_next, E_next, add_to_cache=True):
                continue

            self.frontier.append((e_next, G_next.copy(), E_next.copy(), depth_next))

        # save reward self var
        at_terminal, new_winners = self.at_terminal_state(G, E)

        if at_terminal == -1:
            # not terminal
            self.current_reward = -depth
        elif at_terminal == 1:
            # terminal but no new winner
            self.current_reward = -depth
        elif at_terminal == 2:
            # terminal and found new winner
            self.current_reward = 1
            for c in new_winners:
                assert c not in self.known_winners
                self.known_winners.add(c)

        self.stats.num_nodes += 1

        if not f_testing:
            self.running_nodes += 1

            if self.running_nodes % params.print_loss_every == 0:
                print("*******LOSS:", self.running_loss / params.print_loss_every)
                if self.loss_output_file:
                    self.loss_output_file.write(str(self.running_nodes) + '\t' + str(self.running_loss / params.print_loss_every) + '\n')
                    self.loss_output_file.flush()

                self.running_loss = 0

        return new_winners


    '''
    -depth on all states
    1 if find new winner
    Return reward stored in self var
    '''
    def reward(self):
        return torch.tensor(self.current_reward, dtype = torch.float32)

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


    '''
    test num nodes to finish full PUT-RP search
    '''
    def test_model(self, test_env, begin_time, val_testing):
        self.initialize(test_env)

        nodes_discovered = {}
        runtimes_discovered = {}

        # if doing more than 1 iteration, need to change nodes_discovered and runtimes_discovered to compute average discovery node/time
        assert params.num_test_iterations == 1

        # Sample using model greedily
        # Test with fixed number of iterations
        with torch.no_grad():
            for iter in range(params.num_test_iterations):
                num_nodes = 0
                self.reset_environment()

                while self.at_goal_state()[0] == -1:
                    legal_actions = self.get_legal_actions()

                    num_nodes += 1

                    if val_testing and num_nodes > params.cutoff_testing_nodes:
                        return self.known_winners, nodes_discovered, num_nodes, runtimes_discovered

                    if len(legal_actions) == 0:
                        # Nothing in frontier - full search done
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
                        assert 1==2
                        priorities = []
                        for e in legal_actions:
                            Gc = G.copy()
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

                    new_winners = self.make_move(max_action, f_testing = True)

                    if new_winners is not None:
                        for c in new_winners:
                            runtimes_discovered[c] = time.perf_counter() - begin_time
                            nodes_discovered[c] = num_nodes

        return self.known_winners, nodes_discovered, num_nodes, runtimes_discovered