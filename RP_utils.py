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

import params

def save_model(model, details_str, model_id):
    # Save the model
    model_checkpoint  = str(model_id) + "_" + details_str + "_model.pth.tar"
    torch.save(model.state_dict(), rpconfig.results_path + model_checkpoint)
    print("model saved to", model_checkpoint)

def load_model(model, checkpoint_filename):
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint)
    print("loaded model from", checkpoint_filename)

def edges2string(edges, I):
    m = len(I)
    gstring = list(str(0).zfill(m ** 2))
    for e in edges:
        gstring[(e[0] - min(I)) * m + e[1] - min(I)] = '1'

    return ''.join(gstring)

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

# Prints the total sum of weights off each input feature
def print_feature_weights(model_full):
    # Print weights
    model = model_full.parameters()

    is_bias = -1

    weights = []

    sum_weights_from_node_to_output = []

    # collect all the weights
    for p in model:
        if is_bias == -1:
            prev_layer_len = len(p[0])
            next_layer_len = len(p)
            layer_weights = np.zeros((prev_layer_len, next_layer_len))

            for i in range(next_layer_len):
                weights_to_i = p[i]
                for j in range(prev_layer_len):
                    layer_weights[j][i] = weights_to_i[j]

            weights.append(layer_weights)

            sum_weights_from_node_to_output.append(np.zeros((prev_layer_len,1)))

        is_bias = - is_bias

    for i, layer_weights in reversed(list(enumerate(weights))):
        for j, w in enumerate(layer_weights):
            if i == len(weights) - 1:
                # on last layer
                for k in range(len(w)):
                    sum_weights_from_node_to_output[i][j] += w[k]
            else:
                for k in range(len(w)):
                    sum_weights_from_node_to_output[i][j] += w[k] * sum_weights_from_node_to_output[i+1][k]

    for i in sum_weights_from_node_to_output[0]:
        print(i[0])



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

# Returns array of val to each power from 1 to i
def polynomialize(val, i):
    return [val**j for j in range(1, i+1)]

def safe_div(num, denom):
    if denom == 0:
        return 0
    return num / denom

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