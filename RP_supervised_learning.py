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

class RP_SL():
    def __init__(self):
        # loss_fn = torch.nn.MSELoss(size_average=False)
        self.loss_fn = torch.nn.SmoothL1Loss(size_average=False)  # Huber loss
        # self.loss_fn = torch.nn.HingeEmbeddingLoss() # Hinge loss (but it's not actually)

    def string2edges(self, gstring, I):
        m = len(I)
        edges = []
        for i in range(len(gstring)):
            if gstring[i] == '1':
                e1 = i % m + min(I)
                e0 = int((i - e1) / m) + min(I)
                edges.append((e0, e1))
        return edges

    def string2K(self, K_str):
        K = set()
        for i in range(len(K_str)):
            if K_str[i] == '1':
                K.add(i)
        return K

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

    def safe_div(self, num, denom):
        if denom == 0:
            return 0
        return num / denom

    # Returns array of val to each power from 1 to i
    def polynomialize(self, val, i):
        return [val**j for j in range(1, i+1)]

    def state_features(self, G, K, a, E_0, adjacency_0):
        u = a[0]
        v = a[1]

        f = []

        # out/in degree
        if params.use_in_out:
            f.extend(self.polynomialize(self.safe_div(G.out_degree(u), E_0.out_degree(u)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(G.in_degree(u), E_0.in_degree(u)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(G.out_degree(v), E_0.out_degree(v)), params.num_polynomial))
            f.extend(self.polynomialize(self.safe_div(G.in_degree(v), E_0.in_degree(v)), params.num_polynomial))

        # known winners features
        if params.use_K:
            f.append(2 * int(u in K) - 1)
            f.append(2 * int(v in K) - 1)

        # adjacency matrix if a is added
        if params.use_adjacency_matrix:
            G.add_edge(u,v)
            adjacency = nx.adjacency_matrix(G, nodelist = self.I).todense()
            adjacency = np.multiply(adjacency, adjacency_0)
            adjacency_normalized = np.divide(adjacency, params.n)
            f.extend(adjacency_normalized.flatten().tolist()[0])
            G.remove_edge(u,v)

        # K representation
        if params.use_K_representation:
            K_list = []
            for i in self.I:
                if i in K:
                    K_list.append(1)
                else:
                    K_list.append(0)
            f.extend(K_list)

        # TODO: missing all the rest of the features

        return Variable(torch.from_numpy(np.array(f)).float())

    '''
    Tests number of profiles where highest q val action is correct
    '''
    def test(self, test_data, model, profile_to_E0, profile_to_adjacency0):
        num_correct = 0
        for d in test_data:
            profile = d[0]

            E_0 = profile_to_E0[profile]

            G = nx.DiGraph()
            G.add_nodes_from(self.I)
            G.add_edges_from(self.string2edges(d[1], self.I))
            E = nx.DiGraph()
            E.add_nodes_from(self.I)

            E_edges = self.string2edges(d[2], self.I)
            for e in E_edges:
                E.add_edge(e[0], e[1], weight=E_0[e[0]][e[1]]['weight'])

            K = self.string2K(d[3])

            actions_optimal = self.data_state_to_actions[d]

            adjacency_0 = profile_to_adjacency0[profile]

            # get legal actions at state
            legal_actions = self.get_legal_actions(G, E)

            # find q value action
            max_action = None
            max_action_val = float("-inf")

            for e in legal_actions:
                features = self.state_features(G, K, e, E_0, adjacency_0)
                action_Q_val = model(features)
                if action_Q_val > max_action_val:
                    max_action = e
                    max_action_val = action_Q_val

            # for random action selection testing
            # max_action = legal_actions[random.randint(0, len(legal_actions) - 1)]

            if max_action in set(actions_optimal):
                # we're good
                # print('good')
                num_correct += 1

        return num_correct

    def learn_optimal_action(self, model, G, K, E_0, adjacency_0, a):
        # increase winner
        optimal_action_features = self.state_features(G, K, a, E_0, adjacency_0)
        optimal_action_q_val = model(optimal_action_features)

        if optimal_action_q_val.item() < 1:
            loss = self.loss_fn(optimal_action_q_val, Variable(torch.FloatTensor([1.0])))

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= params.SL_optimal_action_learning_rate * p.grad

            return loss.item()

        return 0

    def learn_bad_action(self, model, G, K, E_0, adjacency_0, a):
        # reduce winner
        bad_action_features = self.state_features(G, K, a, E_0, adjacency_0)
        bad_action_q_val = model(bad_action_features)

        if bad_action_q_val.item() > -1:
            loss = self.loss_fn(bad_action_q_val, Variable(torch.FloatTensor([-1.0])))

            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= params.SL_bad_action_learning_rate * p.grad

            return loss.item()

        return 0

    '''
    Main function to run for supervised learning
    '''
    def RP_SL(self, model, model_id, parameters_file):
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

        profile_to_adjacency0 = {}
        profile_to_E0 = {}

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

                E = nx.DiGraph()
                E.add_nodes_from(self.I)
                for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
                    if wmg[cand1][cand2] > 0:
                        E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])

                profile_to_E0[current_profile] = E.copy()

                adjacency_0 = nx.adjacency_matrix(E, nodelist=self.I).todense()

                profile_to_adjacency0[current_profile] = adjacency_0.copy()

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

        loss_output_file.write("Epoch" + '\t' + "Optimal Action Avg Loss Per Action" + '\t' + "Bad Action Avg Loss Per Action" + '\t' + "Percent Correct" + '\n')
        test_output_file.write("Epoch" + '\t' + "Percent Correct" + '\n')
        loss_output_file.flush()
        test_output_file.flush()

        for epoch in range(params.SL_num_epochs):
            running_loss = 0 # loss for "optimal" actions
            running_loss_2 = 0 # loss for "bad" actions
            num_correct = 0
            num_bad_actions = 0
            num_optimal_actions = 0

            # Test model
            if (epoch % params.SL_test_every == 0 and params.SL_test_at_start):
                test_results = self.test(test_data, model, profile_to_E0, profile_to_adjacency0)
                print("Test", epoch, test_results / len(test_data))
                RP_utils.save_model(model, "SL_" + str(epoch), model_id)
                test_output_file.write(str(epoch) + '\t' + str(test_results / len(test_data)) + '\n')
                test_output_file.flush()

            print("--------------------------")
            print("Starting epoch", epoch)
            epoch_start = time.perf_counter()

            for i in range(params.SL_num_training_data):
                d = data[i]
                profile = d[0]

                E_0 = profile_to_E0[profile]

                G = nx.DiGraph()
                G.add_nodes_from(self.I)
                G.add_edges_from(self.string2edges(d[1], self.I))
                E = nx.DiGraph()
                E.add_nodes_from(self.I)

                E_edges = self.string2edges(d[2], self.I)
                for e in E_edges:
                    E.add_edge(e[0], e[1], weight=E_0[e[0]][e[1]]['weight'])

                K = self.string2K(d[3])

                actions_optimal = self.data_state_to_actions[d]

                adjacency_0 = profile_to_adjacency0[profile]

                # get legal actions at state
                legal_actions = self.get_legal_actions(G, E)

                # sanity check
                for a in actions_optimal:
                    assert a not in G.edges()
                    assert a in E.edges()
                    assert a in legal_actions
                assert len(legal_actions) > 0

                # find max q value action
                max_action = None
                max_action_val = float("-inf")
                for e in legal_actions:
                    features = self.state_features(G, K, e, E_0, adjacency_0)
                    action_Q_val = model(features)
                    if action_Q_val > max_action_val:
                        max_action = e
                        max_action_val = action_Q_val

                if max_action in set(actions_optimal):
                    # selecting correctly
                    # but still want to train for q vals
                    num_correct += 1

                # update correct actions to have q val 1
                for e in actions_optimal:
                    running_loss += self.learn_optimal_action(model, G, K, E_0, adjacency_0, e)
                    num_optimal_actions += 1

                # update all other actions to have q val -1
                for e in legal_actions:
                    if e not in set(actions_optimal):
                        running_loss_2 += self.learn_bad_action(model, G, K, E_0, adjacency_0, e)
                        num_bad_actions += 1

            # compute avg loss per action
            running_loss = running_loss / num_optimal_actions
            running_loss_2 = running_loss_2 / num_bad_actions

            loss_output_file.write(str(epoch) + '\t' + str(running_loss) + '\t' + str(running_loss_2) + '\t' + str(num_correct / params.SL_num_training_data) + '\n')
            loss_output_file.flush()
            print("Finished epoch", epoch)
            print("optimal action loss", running_loss)
            print("bad actions loss", running_loss_2)
            print("percent correct", num_correct / params.SL_num_training_data)
            print("num optimal actions", num_optimal_actions)
            print("num bad actions", num_bad_actions)
            print("time for epoch", time.perf_counter() - epoch_start)

        # Final test
        test_results = self.test(test_data, model, profile_to_E0, profile_to_adjacency0)
        print("Test final", test_results / len(test_data))
        RP_utils.save_model(model, "SL_" + str(params.SL_num_epochs), model_id)
        test_output_file.write(str(params.SL_num_epochs) + '\t' + str(test_results / len(test_data)) + '\n')
        test_output_file.flush()

        loss_output_file.close()
        test_output_file.close()

# reinforcement learning type approach which adjusts optimal action q val to whatever current max is
# spirals out of control
# for epoch in range(num_epochs):
#     running_loss = 0
#     running_loss_2 = 0
#     num_correct = 0
#     for i in range(100):
#         d = data[i]
#         profile = d[0]
#
#         E_0 = profile_to_E0[profile]
#
#         G = nx.DiGraph()
#         G.add_nodes_from(I)
#         G.add_edges_from(string2edges(d[1], I))
#         E = nx.DiGraph()
#         E.add_nodes_from(I)
#
#         E_edges = string2edges(d[2], I)
#         for e in E_edges:
#             E.add_edge(e[0], e[1], weight=E_0[e[0]][e[1]]['weight'])
#
#         K = string2K(d[3])
#         a_optimal = d[4]
#
#         adjacency_0 = profile_to_adjacency0[profile]
#
#         # remove a_optimal from G and add to E
#         G.remove_edges_from([a_optimal])
#         E.add_edge(a_optimal[0], a_optimal[1], weight=E_0[a_optimal[0]][a_optimal[1]]['weight'])
#
#         # get legal actions at state
#         legal_actions = get_legal_actions(G, E)
#
#         # find q value action
#         max_action = None
#         max_action_val = float("-inf")
#
#         assert a_optimal not in G.edges()
#         assert a_optimal in E.edges()
#         assert a_optimal in legal_actions
#         assert len(legal_actions) > 0
#
#         for e in legal_actions:
#             features = state_features(G, K, e, E_0, adjacency_0)
#             action_Q_val = model(features)
#             if action_Q_val > max_action_val:
#                 max_action = e
#                 max_action_val = action_Q_val
#
#         if max_action == a_optimal:
#             # we're good
#             # print('good')
#             num_correct += 1
#             continue
#
#         # compute loss as difference from supervised correct action a to max action q val
#         correct_features = state_features(G, K, a_optimal, E_0, adjacency_0)
#         current_q_val = model(correct_features)
#
#         loss = loss_fn(current_q_val, Variable(max_action_val))
#
#         # update model
#         model.zero_grad()
#         loss.backward()
#         with torch.no_grad():
#             for param in model.parameters():
#                 param -= learning_rate * param.grad
#
#         # reduce winner
#         incorrect_winner_features = state_features(G, K, max_action, E_0, adjacency_0)
#         incorrect_winner_q_val = model(incorrect_winner_features)
#
#         loss2 = loss_fn(incorrect_winner_q_val, current_q_val.detach())
#
#         model.zero_grad()
#         loss2.backward()
#         with torch.no_grad():
#             for param in model.parameters():
#                 param -= learning_rate_2 * param.grad
#
#         running_loss += loss.item()
#         running_loss_2 += loss2.item()
#
#
#     loss_output_file.write(str(running_loss) + '\t' + str(running_loss_2) + '\t' + str(num_correct) + '\n')
#     loss_output_file.flush()
#     print(running_loss)
#     print(running_loss_2)
#     print(num_correct)
#     print(current_q_val.item())
#     print('------------------------------------')




