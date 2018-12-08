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

def string2edges(gstring, I):
    m = len(I)
    edges = []
    for i in range(len(gstring)):
        if gstring[i] == '1':
            e1 = i % m + min(I)
            e0 = int((i - e1) / m) + min(I)
            edges.append((e0, e1))
    return edges

def normalize_probs(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Returns input layer features over all edges at current state G, W
def state_features(G, I, W):
    f = []

    # in/out matrices
    out_degree = G.out_degree(I)
    for (i, j) in out_degree:
        f.append(j / len(I))

    in_degree = G.in_degree(I)
    for (i, j) in in_degree:
        f.append(j / len(I))

    # W representation
    W_list = []
    for i in I:
        if i in W:
            W_list.append(1)
        else:
            W_list.append(0)
    f.extend(W_list)

    return Variable(torch.from_numpy(np.array(f)).float())


if __name__ == '__main__':
    # Set random seeds
    random.seed(time.time())
    torch.manual_seed(time.time())

    os.chdir(rpconfig.path)

    # tunable params
    num_iters = 1000 # num training iterations
    K = 5 # num strategies to sample each iteration
    m = 10 # num candidates
    learning_rate = 0.01

    I = [i for i in range(m)]

    # make policy model
    policy = torch.nn.Sequential(
        torch.nn.Linear(2 * m + m, m * (m - 1)),
        torch.nn.Softmax(dim=0)
    )
    policy.apply(init_weights)

    # load profiles - 14k m10n10
    filenames_file = open(rpconfig.filename_profiles, 'r')
    filenames = [i.strip('\n') for i in filenames_file]
    train_filenames = filenames[:1000] #+ filenames[12000:]
    test_filenames = filenames[10000:11000]  # the same 1000 profiles we used in the paper
    validation_filenames = filenames[11000:11500]

    train_profiles = [read_profile(i) for i in train_filenames]
    test_profiles = [read_profile(i) for i in test_filenames]
    val_profiles = [read_profile(i) for i in validation_filenames]

    # D contains training data accumulated through learning
    # contains tuples ((G,W), (prob distribution over all edges))
    D = set()

    Q = {} # strategies -> num_nodes

    PUT_RP = PUT_RP_using_policy()

    loss_fn = torch.nn.MSELoss(size_average=False, reduce=True)  # using mean squared error

    # order the edges
    edges_ordered = {}
    index = 0
    for i in range(len(I)):
        for j in range(len(I)):
            if i != j:
                edges_ordered[(i, j)] = index
                index += 1


    loss_output_file = open('DAgger_loss.txt', 'w+')
    running_loss = 0
    print_loss_every = 1000
    state_count = 0

    for iter in range(num_iters):
        print("---------------iter " + str(iter) + '-----------------', flush=True)

        torch.save(policy.state_dict(), "DAgger_policy_" + str(iter) + "_model.pth.tar")

        # update policy for exploration
        # TODO

        print("------sampling strategies-----------", flush=True)
        # sample K strategies for each profile in training profiles
        with torch.no_grad():
            for i in range(len(train_profiles)):
                profile = train_profiles[i]
                print("sampling", train_filenames[i], flush=True)
                for j in range(K):
                    strategy, num_nodes, timed_out = PUT_RP.sample_strategy(profile, policy)
                    Q[strategy] = num_nodes
                    print("sampled strategy", j, "with nodes", num_nodes, flush=True)

                    # if not timed_out: # not sure if we'll have luxury to be able to avoid timeout runs
                    #     Q[strategy] = num_nodes

        print("------extracting training data-----------", flush=True)
        # extract training data from strategies Q and store in D
        # R is dict of (G,W) -> dict of e -> best num nodes achieved by selecting edge e from state (G,W)
        R = {}
        for (strategy, num_nodes) in Q.items():
            # strategy is list ((G_str,W), e)
            for (s, e) in strategy:
                # (s,e) = (state, action)
                if s not in R:
                    R[s] = {}
                if e not in R[s]:
                    R[s][e] = num_nodes
                else:
                    R[s][e] = min(num_nodes, R[s][e])

        # using R, add points ((G,W), T) to D where T is prob dist over all possible edges e of 1 / R[(G,W)][e] (or 0 if e is illegal choice)
        # TODO: not sure what to do if (G,W) already in D - should you replace the T for (G,W) or should you have two data points with the same (G,W) and different T?
        for (s, E) in R.items():
            # E is dict of e -> best num nodes achieved
            T = []
            for e in edges_ordered.keys():
                if e in E:
                    T.append(1 / E[e])
                else:
                    T.append(0)

            T = tuple(normalize_probs(T))

            D.add((s, T))

        print("------training-----------", flush=True)
        i = 0
        # train on D
        for (S, T) in D:
            print("training", i, "of", len(D), flush=True)
            i += 1
            # S is state, T is prob distribution over all edges
            G = nx.DiGraph(string2edges(S[0], I))
            G.add_nodes_from(I)
            W = S[1]

            x = policy(state_features(G, I, W))
            y = torch.tensor(T, dtype = torch.float32)

            # Compute loss
            loss = loss_fn(x, y)

            running_loss += loss.item()
            state_count += 1
            if state_count % print_loss_every == 0:
                loss_output_file.write(str(state_count) + '\t' + str(running_loss / print_loss_every) + '\n')
                loss_output_file.flush()
                running_loss = 0

            # Gradient descent

            # Zero the gradients before running the backward pass.
            policy.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor
            with torch.no_grad():
                for param in policy.parameters():
                    param -= learning_rate * param.grad

        print("len Q", len(Q))
        print("len D", len(D))
        print('-----------------------')


    loss_output_file.close()



