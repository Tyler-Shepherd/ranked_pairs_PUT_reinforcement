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
import rpconfig
from collections import defaultdict
import matplotlib.pyplot as plt
from queue import PriorityQueue
import torch
from torch.autograd import Variable
import random
from pprint import pprint
import glob
import datetime

sys.path.append('./RL')
from RL_base import RL_base
from RL_base_experience_replay import RL_base_experience_replay
from RP_RL_agent import RP_RL_agent
from RP_RL_agent_node2vec import RP_RL_agent_node2vec


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

def read_Y_prediction(inputfile):
    Y = dict()
    temp = inputfile.readline()
    filenames = []
    while temp:
        infomation = temp.strip().split(":")
        filenames.append(infomation[0])
        # print(infomation[1])
        x = infomation[1].split()
        # print(x)
        # x = x.split(', ')
        # print(x)
        Y[infomation[0]] = [ float( x ) for x in x if x ]
        # print(Y[infomation[0]])
        temp = inputfile.readline()

    return Y, filenames

def test_model(test_output_file, agent, test_filenames):
    # Test the agents learned model

    num_found_test = 0
    j = 0

    agent.save_model()

    print("Starting test")

    start = time.perf_counter()

    for test_inputfile in test_filenames:
        test_profile = read_profile(test_inputfile)

        # print("Testing", test_inputfile)

        test_PUT_winners, times_discovered = agent.test_model(test_profile, num_test_iterations)

        num_found_test += len(test_PUT_winners)

        j += 1

    print("Test found", num_found_test, "took", time.perf_counter() - start)
    test_output_file.write(str(i) + "\t" + str(num_found_test) + "\n")
    test_output_file.flush()


# After how many profiles to test the model
test_every = 2500

# Whether or not to test before any training
test_at_start = 1

# Number of iterations to use when testing
num_test_iterations = 5

# Whether to initialize model from default values (for comparison purposes)
f_start_from_default = 1

f_experience_replay = 0

# Path to default model (used only if f_start_from_default)
default_model_path = "C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\\\m10n10-100k\\default_agent_7_features.pth.tar"


if __name__ == '__main__':

    # Set random seeds
    random.seed(time.time())
    torch.manual_seed(time.time())

    # Read the filenames
    os.chdir(rpconfig.path)
    y1_filenames = rpconfig.filename
    inf1 = open(y1_filenames, 'r')
    prediction, filenames = read_Y_prediction(inf1)
    inf1.close()

    filenames = sorted(glob.glob('M10N10-*.csv'))

    train_filenames = filenames[0:80000]
    test_filenames = filenames[80000:100000]

    # Open files for output
    output_filename = "results_" + os.path.basename(__file__)
    output_filename = output_filename.replace('.py','')
    weight_filename = output_filename + "_weights.txt"
    loss_filename = output_filename + "_loss.txt"
    test_output_filename = output_filename + "_test_output.txt"
    parameters_output_filename = output_filename + "_parameters.txt"
    output_filename += '.txt'
    output_file = open(output_filename, "w+")
    weight_file = open(weight_filename, "w+")
    test_output_file = open(test_output_filename, "w+")
    parameters_file = open(parameters_output_filename, "w+")
    loss_file = open(loss_filename, "w+")

    # Create base
    if f_experience_replay:
        base = RL_base_experience_replay(len(train_filenames))
    else:
        base = RL_base(len(train_filenames))


    # Create agent
    agent = RP_RL_agent(base.learning_rate, loss_file)
    # agent = RP_RL_agent_node2vec(base.learning_rate)

    if f_start_from_default:
        agent.load_model(default_model_path)

    num_profiles = 0
    total_time = 0

    total_winners_found = 0

    # Counter variable
    i = 0

    # Print header
    header = "Inputfile\tPUT-winners\tExploration Rate\tLearning Rate\tStop Conditions\tNum Nodes\tNum Winners Found\tLoss\tAvg Loss\tIs Acyclic\tRuntime"
    print(header)
    output_file.write(header+'\n')

    # Print parameters
    parameters_file.write("Data Path\t" + rpconfig.path + '\n')
    parameters_file.write("Num Training Data\t" + str(len(train_filenames)) + '\n')
    parameters_file.write("Num Testing Data\t" + str(len(test_filenames)) + '\n')
    parameters_file.write("Train from...to\t" + train_filenames[0] + "\t" + train_filenames[-1] + '\n')
    parameters_file.write("Test from...to\t" + test_filenames[0] + "\t" + test_filenames[-1] + '\n')
    parameters_file.write("Learning Rate\t" + str(base.learning_rate) + '\n')
    parameters_file.write("Learning Rate Decay\t" + str(base.f_learning_rate_decay) + '\n')
    if base.f_learning_rate_decay == 1:
        parameters_file.write("Learning Rate Decay Start\t" + str(base.learning_rate_start) + '\n')
        parameters_file.write("Learning Rate Decay End\t" + str(base.learning_rate_end) + '\n')
        parameters_file.write("Learning Rate Decay Rate\t" + str(base.learning_rate_decay) + '\n')
    parameters_file.write("Discount Factor\t" + str(base.discount_factor) + '\n')
    parameters_file.write("Exploration Rate\t" + str(base.exploration_rate) + '\n')
    parameters_file.write("Exploration Rate Decay\t" + str(base.f_exploration_rate_decay) + '\n')
    if base.f_exploration_rate_decay:
        parameters_file.write("Exploration Rate Decay Start\t" + str(base.exploration_rate_start) + '\n')
        parameters_file.write("Exploration Rate Decay End\t" + str(base.exploration_rate_end) + '\n')
        parameters_file.write("Exploration Rate Decay Rate\t" + str(base.exploration_rate_decay) + '\n')
    parameters_file.write("Num Iterations per Profile\t" + str(base.num_iterations) + '\n')
    parameters_file.write("Agent D_in\t" + str(agent.D_in) + '\n')
    parameters_file.write("Agent H\t" + str(agent.H) + '\n')
    parameters_file.write("Agent D_out\t" + str(agent.D_out) + '\n')
    parameters_file.write("Agent Model\t" + str(agent.model) + '\n')
    parameters_file.write("Agent Loss Function\t" + str(agent.loss_fn) + '\n')
    parameters_file.write("Agent Optimizer Type\t" + str(agent.optimizer_type) + '\n')
    if f_start_from_default:
        parameters_file.write("Default Model File\t" + default_model_path + '\n')
    else:
        parameters_file.write("Default Model File\tN/A\n")
    parameters_file.write("Experience Replay\t" + str(f_experience_replay) + '\n')
    if f_experience_replay:
        parameters_file.write("Experience Replay Buffer Size\t" + str(base.buffer_size) + '\n')
        parameters_file.write("Experience Replay Sample Factor\t" + str(base.unusual_sample_factor) + '\n')
        parameters_file.write("Experience Replay Batch Size\t" + str(base.batch_size) + '\n')
        parameters_file.write("Experience Replay Train Every\t" + str(base.train_every_iterations) + '\n')

    parameters_file.write("Date\t" + str(datetime.datetime.now()) + '\n')

    parameters_file.flush()

    for inputfile in train_filenames:
        if i % test_every == 0 and (test_at_start or i != 0):
            test_model(test_output_file, agent, test_filenames)

        profile = read_profile(inputfile)

        # Run the profile
        start = time.perf_counter()
        rp_results = base.reinforcement_loop(agent, profile)
        end = time.perf_counter()

        PUT_winners = sorted(rp_results.known_winners)
        stats = agent.stats

        total_time += (end - start)
        num_profiles += 1

        if stats.num_nodes == 0:
            avg_loss_per_node = 0
        else:
            avg_loss_per_node = stats.running_loss / stats.num_nodes

        is_acyclic = str(nx.is_directed_acyclic_graph(agent.E_0))

        result_text = "%s\t%r\t%f\t%f\t%r\t%d\t%d\t%f\t%f\t%s\t%f" % (inputfile, PUT_winners, base.exploration_rate, base.learning_rate, stats.stop_condition_hits, stats.num_nodes, len(PUT_winners), stats.running_loss, avg_loss_per_node, is_acyclic, end - start)
        print(i, result_text)
        output_file.write(result_text + '\n')
        output_file.flush()

        agent.print_model(weight_file, f_print_to_console=False)

        i += 1

    # Final test
    test_model(test_output_file, agent, test_filenames)

    # Close files
    output_file.close()
    weight_file.close()
    test_output_file.close()
    parameters_file.close()

    print("Total Time to Train: %f" % total_time)
    print("Average Time: %f" % (total_time / num_profiles))