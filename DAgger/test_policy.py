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

# based off http://papers.nips.cc/paper/8233-learning-to-solve-smt-formulas.pdf and https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf
# Learns to solve PUT-RP in as few expanded nodes as possible using the DAgger learning algorithm

from PUT_RP_using_policy import PUT_RP_using_policy

def read_profile(inputfile):
    inf = open(inputfile, 'r')
    cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
    inf.close()

    profile = Profile(cmap, preferences=[])
    Profile.importPreflibFile(profile, inputfile)

    # Currently, we expect the profile to contain complete ordering over candidates. Ties are allowed however.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "soi" and elecType != "csv":
        print("ERROR: unsupported election type")
        exit()

    return profile

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# Use to easily test some learned policy

if __name__ == '__main__':
    # Set random seeds
    random.seed(time.time())
    torch.manual_seed(time.time())

    os.chdir(rpconfig.path)

    m = 10 # num candidates
    I = [i for i in range(m)]

    # make policy model
    policy = torch.nn.Sequential(
        torch.nn.Linear(2 * m + m, m * (m - 1)),
        torch.nn.Softmax(dim=0)
    )
    policy.apply(init_weights)

    # *** EDIT THIS LINE TO CHANGE WHICH POLICY TO TEST ***
    # checkpoint_filename = ""
    # checkpoint = torch.load(checkpoint_filename)
    # policy.load_state_dict(checkpoint)
    # print("loaded model from", checkpoint_filename)

    # load profiles - 14k m10n10
    filenames_file = open(rpconfig.filename_profiles, 'r')
    filenames = [i.strip('\n') for i in filenames_file]
    test_filenames = filenames[10000:11000]  # the same 1000 profiles we used in the paper
    test_profiles = [read_profile(i) for i in test_filenames]

    # Read true winners
    os.chdir(rpconfig.winners_path)
    true_winners = []
    winners_file = open("./winners_14k.txt", 'r')
    for line in winners_file:
        winners = []
        line = line.replace('[', '')
        line = line.replace(']', '')
        line = line.replace(' ', '')
        line = line.replace('\n', '')
        line = line.split(',')
        for c in line:
            winners.append(int(c))
        true_winners.append(winners)
    os.chdir(rpconfig.path)

    true_winners_test = true_winners[10000:11000]

    assert (len(true_winners_test) == len(test_filenames))

    PUT_RP = PUT_RP_using_policy()

    test_output_file = open('test_policy.txt', 'w+')
    test_output_file.write('Filename\tNum Nodes\t100% Nodes\tRuntime\t100% Time\n')
    test_output_file.flush()

    total_num_nodes = 0
    total_nodes_100 = 0
    total_runtime = 0
    total_time_100 = 0
    with torch.no_grad():
        for i in range(len(test_profiles)):
            profile = test_profiles[i]
            profile_winners = true_winners_test[i]
            print("testing", test_filenames[i], flush=True)
            start_test = time.perf_counter()
            stats, found_winners = PUT_RP.test_policy(profile, policy)
            test_time = time.perf_counter() - start_test
            num_nodes = stats.num_nodes
            nodes_100 = max(stats.discovery_states.values())
            time_100 = max(stats.discovery_times.values())
            print("results: nodes", num_nodes, "100% nodes", nodes_100, "runtime", test_time, "100% runtime", time_100, flush=True)

            assert set(found_winners) == set(profile_winners)
            test_output_file.write(str(test_filenames[i]) + '\t' + str(num_nodes) + '\t' + str(nodes_100) + '\t' + str(test_time) + '\t' + str(time_100) + '\n')
            test_output_file.flush()
            total_num_nodes += num_nodes
            total_nodes_100 += nodes_100
            total_runtime += test_time
            total_time_100 += time_100

    n = len(test_profiles)
    print("Avg num nodes", total_num_nodes / n)
    print("Avg 100% nodes", total_nodes_100 / n)
    print("Avg runtime", total_runtime / n)
    print("Avg 100% runtime", total_time_100 / n)
    test_output_file.flush()
    test_output_file.close()



