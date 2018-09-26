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


def string2edges(gstring, I):
    m = len(I)
    edges = []
    for i in range(len(gstring)):
        if gstring[i] == '1':
            e1 = i % m + min(I)
            e0 = int((i - e1) / m) + min(I)
            edges.append((e0, e1))
    return edges

def string2K(K_str):
    K = set()
    for i in range(len(K_str)):
        if K_str[i] == '1':
            K.add(i)
    return K

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

'''
Returns the highest weight edges in E that do not cause a cycle if added
'''
def get_legal_actions(G, E):
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



def state_features(G, K, a, adjacency_0):
    u = a[0]
    v = a[1]

    f = []

    f.extend([1, 1, 1, 1])

    G.add_edge(u,v)
    adjacency = nx.adjacency_matrix(G, nodelist = I).todense()
    adjacency = np.multiply(adjacency, adjacency_0)
    adjacency_normalized = np.divide(adjacency, 10) # NOTE: update if not using n10
    f.extend(adjacency_normalized.flatten().tolist()[0])
    G.remove_edge(u,v)

    # K representation
    K_list = []
    for i in I:
        if i in K:
            K_list.append(1)
        else:
            K_list.append(0)
    f.extend(K_list)

    return Variable(torch.from_numpy(np.array(f)).float())


if __name__ == '__main__':

    data_file = open('missed_winners_data.dat', 'r')

    os.chdir(rpconfig.path)

    # need to update for anything other than m10
    I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    data = []
    current_profile = None

    profile_to_adjacency0 = {}
    profile_to_E0 = {}

    n = 0

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
            E.add_nodes_from(I)
            for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
                if wmg[cand1][cand2] > 0:
                    E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])

            profile_to_E0[current_profile] = E.copy()

            adjacency_0 = nx.adjacency_matrix(E, nodelist=I).todense()

            profile_to_adjacency0[current_profile] = adjacency_0.copy()

        if len(line) == 5:
            # each line in form G E K a[0] a[1]
            G_str = line[0]
            E_str = line[1]
            K_str = line[2]
            a = (int(line[3]), int(line[4]))

            data.append((current_profile, G_str, E_str, K_str, a))

            n += 1

    print("n", n)

    # supervised learning
    D_in = 114
    H = 10000  # first hidden dimension
    D_out = 1  # output dimension, just want q value

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H, D_out)
    )

    model.apply(init_weights)

    loss_fn = torch.nn.MSELoss(size_average=False)
    # loss_fn = torch.nn.SmoothL1Loss(size_average=False)  # Huber loss

    learning_rate = 0.05

    loss_output_file = open('RP_supervised_learning_loss.txt', 'w')

    for d in data:
        profile = d[0]

        E_0 = profile_to_E0[profile]

        G = nx.DiGraph()
        G.add_nodes_from(I)
        G.add_edges_from(string2edges(d[1], I))
        E = nx.DiGraph()
        E.add_nodes_from(I)

        E_edges = string2edges(d[2], I)
        for e in E_edges:
            E.add_edge(e[0], e[1], weight=E_0[e[0]][e[1]]['weight'])

        K = string2K(d[3])
        a_optimal = d[4]

        adjacency_0 = profile_to_adjacency0[profile]

        # get legal actions at state
        legal_actions = get_legal_actions(G, E)

        # find q value action
        max_action = None
        max_action_val = float("-inf")

        assert a_optimal not in G.edges()
        assert a_optimal in E.edges()
        assert len(legal_actions) != 0

        for e in legal_actions:
            features = state_features(G, K, e, adjacency_0)
            action_Q_val = model(features)
            if action_Q_val > max_action_val:
                max_action = e
                max_action_val = action_Q_val

        # compute loss as difference from supervised correct action a to max action q val
        correct_features = state_features(G, K, a_optimal, adjacency_0)
        current_q_val = model(correct_features)

        loss = loss_fn(current_q_val, Variable(max_action_val))

        # update model
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        loss_output_file.write(str(loss.item()) + '\n')
        print(loss.item())

    loss_output_file.close()