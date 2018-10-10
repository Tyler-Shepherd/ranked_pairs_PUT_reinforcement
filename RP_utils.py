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