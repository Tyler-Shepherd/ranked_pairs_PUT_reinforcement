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



def test_policy(test_profiles, test_filenames, true_winners_test, policy, output_file):
    total_num_nodes = 0
    total_nodes_100 = 0
    total_runtime = 0
    total_time_100 = 0
    with torch.no_grad():
        for i in range(len(test_profiles)):
            profile = test_profiles[i]
            profile_winners = set(true_winners_test[i])
            print("Testing", test_filenames[i], flush=True)
            start_test = time.perf_counter()
            stats, found_winners = PUT_RP.test_policy(profile, policy)
            test_time = time.perf_counter() - start_test
            num_nodes = stats.num_nodes
            nodes_100 = max(stats.discovery_states.values())
            time_100 = max(stats.discovery_times.values())
            print("Results: nodes", num_nodes, "100% nodes", nodes_100, "runtime", test_time, "100% runtime", time_100, flush=True)

            assert found_winners == profile_winners

            output_file.write(str(test_filenames[i]) + '\t' + str(num_nodes) + '\t' + str(nodes_100) + '\t' + str(test_time) + '\t' + str(time_100) + '\n')
            output_file.flush()
            total_num_nodes += num_nodes
            total_nodes_100 += nodes_100
            total_runtime += test_time
            total_time_100 += time_100

    output_file.write('------------------------------------------\n')
    output_file.flush()

    n = len(test_profiles)
    print("Avg nodes", total_num_nodes / n)
    print("Avg 100% nodes", total_nodes_100 / n)
    print("Avg runtime", total_runtime / n)
    print("Avg 100% runtime", total_time_100 / n)

    return total_num_nodes / n, total_nodes_100 / n, total_runtime / n, total_time_100 / n

if __name__ == '__main__':
    # Set random seeds
    random.seed(time.time())
    torch.manual_seed(time.time())

    os.chdir(rpconfig.path)

    # tunable params
    num_iters = 10 # num training iterations
    K = 3 # num strategies to sample each iteration
    m = 10 # num candidates
    learning_rate = 0.01
    exploration_p = 0.2 # at iteration i, strategy sampling explores with prob exploration_p ^ i (so first iteration with i = 0 is pure exploration)
    TIMEOUT_PENALTY = 100000 # "reward" if profile times out during training (larger is worse)
    # TODO: potential idea - instead of giving a single TIMEOUT_PENALTY value, use the number of winners found in penalty
    # - that way a strategy that timed out but found more winners would give better reward than a different strategy that timed out but found less winners

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
    val_filenames = filenames[11000:11100]

    train_profiles = [read_profile(i) for i in train_filenames]
    test_profiles = [read_profile(i) for i in test_filenames]
    val_profiles = [read_profile(i) for i in val_filenames]

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

    os.chdir(os.path.dirname(__file__))

    # Split true_winners into train and test
    true_winners_train = true_winners[:1000] #+ true_winners[12000:]
    true_winners_test = true_winners[10000:11000]
    true_winners_val = true_winners[11000:11100]

    assert(len(true_winners_train) == len(train_filenames))
    assert (len(true_winners_test) == len(test_filenames))
    assert (len(true_winners_val) == len(val_filenames))

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

    val_output_file = open('DAgger_val.txt', 'w+')
    val_output_file.write('filename\tnum nodes\n')
    val_output_summary_file = open('DAgger_val_summary.txt', 'w+')
    val_output_summary_file.write('Iter\tAvg Num Nodes\tAvg 100% Nodes\tAvg Runtime\tAvg 100% Time\n')
    val_output_file.flush()
    val_output_summary_file.flush()
    val_results = []

    loss_output_file = open('DAgger_loss.txt', 'w+')
    running_loss = 0
    print_loss_every = 1000
    state_count = 0

    for iter in range(num_iters):
        print("---------------Iteration " + str(iter) + '-----------------', flush=True)

        # update exploration rate beta
        beta = np.power(exploration_p, iter)

        print("------Sampling Strategies-----------", flush=True)
        # sample K strategies for each profile in training profiles
        with torch.no_grad():
            for i in range(len(train_profiles)):
                profile = train_profiles[i]
                profile_winners = set(true_winners_train[i])
                print("sampling", train_filenames[i], flush=True)
                for j in range(K):
                    start_sample = time.perf_counter()
                    strategy, stats, timed_out, found_winners = PUT_RP.sample_strategy(profile, policy, profile_winners, beta)
                    sample_runtime = time.perf_counter() - start_sample
                    if timed_out:
                        Q[strategy] = TIMEOUT_PENALTY
                    else:
                        Q[strategy] = stats.num_nodes
                    print("sampled strategy", j, "with nodes", stats.num_nodes, "timeout", timed_out, "runtime", sample_runtime, flush=True)

                    if timed_out:
                        assert found_winners <= profile_winners
                    else:
                        assert found_winners == profile_winners

        print("------Extracting Training Data-----------", flush=True)
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

        R = {}

        print("------Training Policy-----------", flush=True)
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

        D = set(random.sample(D, 100000))

        torch.save(policy.state_dict(), "DAgger_policy_" + str(iter) + "_model.pth.tar")

        # validation testing
        print('------------------Validation Testing-------------------')

        avg_num_nodes, avg_100_nodes, avg_runtime, avg_100_time = test_policy(val_profiles, val_filenames, true_winners_val, policy, val_output_file)
        val_results.append(avg_num_nodes)
        val_output_summary_file.write(str(iter) + '\t' + str(avg_num_nodes) + '\t' + str(avg_100_nodes) + '\t' + str(avg_runtime) + '\t' + str(avg_100_time) + '\n')
        print('-----------------------')

    print(val_results)
    best_model = np.argmin(val_results)
    print("best model:", best_model)

    # test best model from validation data on test data
    test_output_file = open('DAgger_test.txt', 'w+')
    test_output_file.write('Filename\tNum Nodes\t100% Nodes\tRuntime\t100% Time\n')
    test_output_file.flush()

    print("--------Final Test---------------")
    checkpoint = torch.load("DAgger_policy_" + str(best_model) + "_model.pth.tar")
    policy.load_state_dict(checkpoint)

    test_policy(test_profiles, test_filenames, true_winners_test, policy, test_output_file)

    loss_output_file.close()
    val_output_file.close()
    val_output_summary_file.close()
    test_output_file.close()


