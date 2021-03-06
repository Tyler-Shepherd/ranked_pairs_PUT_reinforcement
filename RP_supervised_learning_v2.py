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

import params as params
import RP_utils

class RP_SL_v2():
    def __init__(self):
        # loss_fn = torch.nn.MSELoss(size_average=False)
        self.loss_fn = torch.nn.SmoothL1Loss(size_average=False)  # Huber loss
        # self.loss_fn = torch.nn.HingeEmbeddingLoss() # Hinge loss (but it's not actually)

    '''
    Returns the highest weight edges in E that do not cause a cycle if added
    '''
    def get_legal_actions(self, G, E):
        if len(E.edges()) == 0:
            # There are no possible actions
            return []

        Gc = G.copy()
        G_transitive_closure = nx.transitive_closure(G)

        max_weight = max([(d['weight']) for (u, v, d) in E.edges(data=True)])
        T = [(u, v) for (u, v, d) in E.edges(data=True) if d['weight'] == max_weight]

        legal_actions = []

        for e in T:
            if not G_transitive_closure.has_edge(e[1], e[0]):
                Gc.add_edges_from([e])
                legal_actions.append(e)
                Gc.remove_edges_from([e])

        return legal_actions

    def state_features(self, G, K, T, profile):
        f = []

        E_0 = self.profile_to_E0[profile]
        adjacency_0 = self.profile_to_adjacency0[profile]

        if params.use_in_out_matrix:
            out_degree = G.out_degree(self.I)
            for (i,j) in out_degree:
                f.extend(RP_utils.polynomialize(RP_utils.safe_div(j, E_0.out_degree(i)), params.num_polynomial))

            in_degree = G.in_degree(self.I)
            for (i,j) in in_degree:
                f.extend(RP_utils.polynomialize(RP_utils.safe_div(j, E_0.in_degree(i)), params.num_polynomial))

        if params.use_total_degree_matrix:
            for i in self.I:
                i_total = G.out_degree(i) + G.in_degree(i)
                i_e0_total = E_0.out_degree(i) + E_0.in_degree(i)
                f.extend(RP_utils.polynomialize(RP_utils.safe_div(i_total, i_e0_total), params.num_polynomial))

        if params.use_in_out_binary_matrix:
            out_degree = G.out_degree(self.I)
            for (i,j) in out_degree:
                f.append(2 * int(j > 0) - 1)

            in_degree = G.in_degree(self.I)
            for (i,j) in in_degree:
                f.append(2 * int(j > 0) - 1)

        if params.use_voting_rules_matrix:
            for i in self.profile_to_plurality[profile]:
                f.extend(RP_utils.polynomialize(i, params.num_polynomial))
            for i in self.profile_to_borda[profile]:
                f.extend(RP_utils.polynomialize(i, params.num_polynomial))
            for i in self.profile_to_copeland[profile]:
                f.extend(RP_utils.polynomialize(i, params.num_polynomial))
            for i in self.profile_to_maximin[profile]:
                f.extend(RP_utils.polynomialize(i, params.num_polynomial))

        if params.use_edge_weight:
            f.extend(RP_utils.polynomialize(E_0[T[0][0]][T[0][1]]['weight'] / self.profile_to_max_edge_weight[profile], params.num_polynomial))

        if params.use_vectorized_wmg:
            f.extend(self.profile_to_vectorized_wmg[profile])

        if params.use_posmat:
            f.extend(self.profile_to_posmat[profile])

        if params.use_tier_adjacency_matrix:
            T_matrix = np.zeros((int(params.m), int(params.m)))
            for (c1,c2) in T:
                T_matrix[c1,c2] = 1
            T_vec = list(T_matrix.flatten())
            f.extend(T_vec)

        if params.use_connectivity_matrix:
            for i in self.I:
                for j in self.I:
                    if i != j:
                        f.extend(RP_utils.polynomialize(local_edge_connectivity(G, i, j) / (params.m-2), params.num_polynomial)) # normalized by m-2 since max edges needed to disconnect i and j is all edges but i -> i and i -> j
            all_pairs_node_connectivity = nx.all_pairs_node_connectivity(G)
            for i in self.I:
                for j in self.I:
                    if i != j:
                        f.extend(RP_utils.polynomialize(all_pairs_node_connectivity[i][j] / (params.m-2), params.num_polynomial)) # normalized by m-2 since max nodes needed to disconnect i and j is all nodes but i and j

        # adjacency matrix
        if params.use_adjacency_matrix:
            adjacency = nx.adjacency_matrix(G, nodelist = self.I).todense()
            adjacency = np.multiply(adjacency, adjacency_0)
            adjacency_normalized = np.divide(adjacency, params.n)
            f.extend(adjacency_normalized.flatten().tolist()[0])

        # K representation
        if params.use_K_representation:
            K_list = []
            for i in self.I:
                if i in K:
                    K_list.append(1)
                else:
                    K_list.append(0)
            f.extend(K_list)

        return Variable(torch.from_numpy(np.array(f)).float())

    '''
    Tests number of profiles where highest q val action is correct
    '''
    def test(self, test_data, model):
        num_correct = 0
        for d in test_data:
            profile = d[0]

            E_0 = self.profile_to_E0[profile]

            G = nx.DiGraph()
            G.add_nodes_from(self.I)
            G.add_edges_from(RP_utils.string2edges(d[1], self.I))
            E = nx.DiGraph()
            E.add_nodes_from(self.I)

            E_edges = RP_utils.string2edges(d[2], self.I)
            for e in E_edges:
                E.add_edge(e[0], e[1], weight=E_0[e[0]][e[1]]['weight'])

            K = RP_utils.string2K(d[3])

            actions_optimal = self.data_state_to_actions[d]

            # get legal actions at state
            legal_actions = self.get_legal_actions(G, E)

            # find max q value action
            action_Q_vals = model(self.state_features(G, K, legal_actions, profile))
            max_action = None
            max_action_val = float("-inf")
            for e in legal_actions:
                action_Q_val = action_Q_vals[self.edges_ordered[e]]
                if action_Q_val > max_action_val:
                    max_action = e
                    max_action_val = action_Q_val

            assert max_action is not None

            if max_action in set(actions_optimal):
                # we're good
                # print('good')
                num_correct += 1

        return num_correct

    '''
    Main function to run for supervised learning
    '''
    def RP_SL_v2(self, model, model_id, parameters_file):
        print("***********************************************")
        print("Starting Supervised Learning", model_id)

        parameters_file.write("SL Loss Function\t" + str(self.loss_fn) + '\n')
        parameters_file.flush()

        data_file = open('missed_winners_data.dat', 'r')
        os.chdir(rpconfig.path)

        self.I = [i for i in range(int(params.m))]

        # dict of tuples (profile, G, E, K) -> list of actions (u,v)
        self.data_state_to_actions = {}
        current_profile = None

        self.profile_to_adjacency0 = {}
        self.profile_to_E0 = {}
        self.profile_to_plurality = {}
        self.profile_to_borda = {}
        self.profile_to_copeland = {}
        self.profile_to_maximin = {}
        self.profile_to_max_edge_weight = {}
        self.profile_to_vectorized_wmg = {}
        self.profile_to_posmat = {}

        n = 0
        n2 = 0

        # read data
        for line in data_file:
            line = line.strip('\n')
            line = line.split('\t')

            if len(line) == 1:
                current_profile = line[0]

                # read profile
                inf = open(current_profile, 'r')
                cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
                inf.close()

                profile = Profile(cmap, preferences=[])
                Profile.importPreflibFile(profile, current_profile)

                wmg = profile.getWmg()
                profile_matrix = []
                for p in profile.preferences:
                    profile_matrix.append(p.getOrderVector())
                profile_matrix = np.asmatrix(profile_matrix)

                E = nx.DiGraph()
                E.add_nodes_from(self.I)
                self.profile_to_max_edge_weight[current_profile] = 0
                for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
                    if wmg[cand1][cand2] > 0:
                        E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])
                        self.profile_to_max_edge_weight[current_profile] = max(self.profile_to_max_edge_weight[current_profile], wmg[cand1][cand2])

                self.profile_to_E0[current_profile] = E.copy()

                adjacency_0 = nx.adjacency_matrix(E, nodelist=self.I).todense()
                self.profile_to_adjacency0[current_profile] = adjacency_0.copy()

                # compute voting rules scores
                self.profile_to_plurality[current_profile] = RP_utils.plurality_score(profile_matrix)
                self.profile_to_borda[current_profile] = RP_utils.borda_score(profile_matrix)
                self.profile_to_copeland[current_profile] = RP_utils.copeland_score(wmg)
                self.profile_to_maximin[current_profile] = RP_utils.maximin_score(wmg)

                self.profile_to_vectorized_wmg[current_profile] = RP_utils.vectorize_wmg(wmg)
                self.profile_to_posmat[current_profile] = RP_utils.profile2posmat(profile_matrix)

            if len(line) == 5:
                # each line in form G E K a[0] a[1]
                G_str = line[0]
                E_str = line[1]
                K_str = line[2]
                a = (int(line[3]), int(line[4]))

                data_key = (current_profile, G_str, E_str, K_str)

                if data_key in self.data_state_to_actions:
                    self.data_state_to_actions[data_key].append(a)
                else:
                    self.data_state_to_actions[data_key] = [a]
                    n2 += 1

                n += 1

        print("total data", n)
        print("unique states", n2)
        print("have same state", n-n2)

        data = list(self.data_state_to_actions.keys())
        random.shuffle(data)

        test_data = data[-params.SL_num_test_data:]

        # Open files for output
        loss_output_file = open(rpconfig.results_path + str(model_id) + '_SL_loss.txt', 'w+')
        test_output_file = open(rpconfig.results_path + str(model_id) + '_SL_test_results.txt', 'w+')

        loss_output_file.write("Epoch" + '\t' + "Avg Loss Per State" + '\t' + "Percent Correct" + '\n')
        test_output_file.write("Epoch" + '\t' + "Percent Correct" + '\t' + "Time to Test" + '\n')
        loss_output_file.flush()
        test_output_file.flush()

        # order the edges for model ease
        self.edges_ordered = {}
        index = 0
        for i in range(len(self.I)):
            for j in range(len(self.I)):
                if i != j:
                    self.edges_ordered[(i,j)] = index
                    index += 1

        for epoch in range(params.SL_num_epochs):
            running_loss = 0
            num_correct = 0

            # Test model
            if (epoch % params.SL_test_every == 0 and params.SL_test_at_start):
                test_start = time.perf_counter()
                test_results = self.test(test_data, model)
                time_to_test = time.perf_counter() - test_start
                print("Test", epoch, test_results / len(test_data), time_to_test)
                RP_utils.save_model(model, "SL_" + str(epoch), model_id)
                test_output_file.write(str(epoch) + '\t' + str(test_results / len(test_data)) + '\t' + str(time_to_test) + '\n')
                test_output_file.flush()

            print("--------------------------")
            print("Starting epoch", epoch)
            epoch_start = time.perf_counter()

            for i in range(params.SL_num_training_data):
                d = data[i]
                profile = d[0]

                E_0 = self.profile_to_E0[profile]

                G = nx.DiGraph()
                G.add_nodes_from(self.I)
                G.add_edges_from(RP_utils.string2edges(d[1], self.I))
                E = nx.DiGraph()
                E.add_nodes_from(self.I)

                E_edges = RP_utils.string2edges(d[2], self.I)
                for e in E_edges:
                    E.add_edge(e[0], e[1], weight=E_0[e[0]][e[1]]['weight'])

                K = RP_utils.string2K(d[3])

                actions_optimal = self.data_state_to_actions[d]

                # get legal actions at state
                legal_actions = self.get_legal_actions(G, E)

                # sanity check
                for a in actions_optimal:
                    assert a not in G.edges()
                    assert a in E.edges()
                    assert a in legal_actions
                assert len(legal_actions) > 0

                # find max q value action
                action_Q_vals = model(self.state_features(G, K, legal_actions, profile))
                max_action = None
                max_action_val = float("-inf")
                for e in legal_actions:
                    action_Q_val = action_Q_vals[self.edges_ordered[e]]
                    if action_Q_val > max_action_val:
                        max_action = e
                        max_action_val = action_Q_val

                if max_action in set(actions_optimal):
                    # selecting correctly
                    # but still want to train for q vals
                    num_correct += 1

                # update action q vals (correct actions to 1, incorrect actions to -1)
                y = [0] * params.D_out
                for e in actions_optimal:
                    y[self.edges_ordered[e]] = 1

                for e in legal_actions:
                    if e not in set(actions_optimal):
                        y[self.edges_ordered[e]] = -1

                loss = self.loss_fn(action_Q_vals, Variable(torch.FloatTensor(y)))

                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for p in model.parameters():
                        p -= params.SL_optimal_action_learning_rate * p.grad

                running_loss += loss.item()

            # compute avg loss per action
            running_loss = running_loss / params.SL_num_training_data

            time_for_epoch = time.perf_counter() - epoch_start
            loss_output_file.write(str(epoch) + '\t' + str(running_loss) + '\t' + str(num_correct / params.SL_num_training_data) + '\t' + str(time_for_epoch) + '\n')
            loss_output_file.flush()
            print("Finished epoch", epoch)
            print("avg loss", running_loss)
            print("percent correct", num_correct / params.SL_num_training_data)
            print("time for epoch", time_for_epoch)

        # Final test
        test_start = time.perf_counter()
        test_results = self.test(test_data, model)
        time_to_test = time.perf_counter() - test_start
        print("Test final", test_results / len(test_data), time_to_test)
        RP_utils.save_model(model, "SL_" + str(params.SL_num_epochs), model_id)
        test_output_file.write(str(params.SL_num_epochs) + '\t' + str(test_results / len(test_data)) + '\t' + str(time_to_test) + '\n')
        test_output_file.flush()

        loss_output_file.close()
        test_output_file.close()
