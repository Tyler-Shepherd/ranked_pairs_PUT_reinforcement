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
import RP_utils

# Cleans up data from missed_winners_data.dat
# Just manually rename the output missed_winners_clean_data.dat file to missed_winners_data.dat


if __name__ == '__main__':
    data_file = open('missed_winners_data.dat', 'r')
    output_data_file = open('missed_winners_clean_data.dat', 'w+')

    I = [i for i in range(int(params.m))]

    current_profile = None

    current_profile_data = set()

    n_old = 0
    n_new = 0

    # remove duplicate data points
    for line in data_file:
        line = line.strip('\n')
        line = line.split('\t')

        if len(line) == 1:
            current_profile_data = set()
            output_data_file.write(line[0] + '\n')

        elif len(line) == 2:
            output_data_file.write(line[0] + '\t' + line[1] + '\n')

        elif len(line) == 5:
            # each line in form G E K a[0] a[1]
            G_str = line[0]
            E_str = line[1]
            K_str = line[2]
            a = (int(line[3]), int(line[4]))

            new_datapoint = (current_profile, G_str, E_str, K_str, a)

            n_old += 1

            if new_datapoint in current_profile_data:
                continue

            current_profile_data.add(new_datapoint)

            # remove a from G and add back to E because I wrote get_paths_to_missed_winners wrong initially
            G = nx.DiGraph(RP_utils.string2edges(G_str, I))
            E = nx.DiGraph(RP_utils.string2edges(E_str, I))

            G.remove_edges_from([a])
            E.add_edges_from([a])

            G_str = RP_utils.edges2string(G.edges(), I)
            E_str = RP_utils.edges2string(E.edges(), I)

            output_data_file.write(G_str + '\t' + E_str + '\t' + K_str + '\t' + line[3] + '\t' + line[4] + '\n')

            n_new += 1

    print("n old", n_old)
    print("n new", n_new)
    print("removed", n_old - n_new)