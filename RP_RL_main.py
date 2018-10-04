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

from PUT_RP_using_model import MechanismRankedPairs
from PUT_RP_using_model_v2 import MechanismRankedPairs_v2

sys.path.append('./RL')
from RL_base import RL_base
from RL_base_experience_replay import RL_base_experience_replay
from RP_RL_agent import RP_RL_agent
from RP_RL_agent_node2vec import RP_RL_agent_node2vec
from RL_base_v2 import RL_base_v2
from RP_RL_agent_v2 import RP_RL_agent_v2


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

def test_model_find_all_winners(test_output_file, agent, test_filenames, true_winners, model_id, num_times_tested):
    agent.save_model(str(model_id) + "_test_" + str(num_times_tested))

    print("Starting test")
    test_output_file.write('***********************************\n')

    start_test = time.perf_counter()

    for test_inputfile in test_filenames:
        test_profile = read_profile(test_inputfile)

        start = time.perf_counter()
        if f_use_v2:
            rp_results = MechanismRankedPairs_v2().outer_loop_lp(test_profile, agent.model)
        else:
            rp_results = MechanismRankedPairs().outer_loop_lp(test_profile, agent.model)
        end = time.perf_counter()

        PUT_winners = rp_results[0]
        stats = rp_results[1]

        if len(stats.discovery_states.values()) > 0:
            max_discovery_state = max(stats.discovery_states.values())
        else:
            max_discovery_state = -1

        if len(stats.discovery_times.values()) > 0:
            max_discovery_time = max(stats.discovery_times.values())
        else:
            max_discovery_time = - 1

        num_stop_condition_hits = sum(list(stats.stop_condition_hits.values()))

        result_text = "%s\t%r\t%d\t%r\t%d\t%r\t%f\t%r\t%d\t%d\t%d\t%d\t%f\t%f" % (test_inputfile, PUT_winners, stats.num_nodes, stats.discovery_states,
                                                  max_discovery_state, stats.discovery_times, max_discovery_time, stats.stop_condition_hits, num_stop_condition_hits, stats.num_hashes, stats.num_initial_bridges, stats.num_redundant_edges, stats.time_for_cycles, (end - start))

        # print(result_text)
        test_output_file.write(result_text + '\n')
        test_output_file.flush()

    print("Time to test:", time.perf_counter() - start_test)



def test_model(test_output_file, agent, test_filenames, true_winners, model_id):
    # Test the agents learned model

    num_found_test = 0
    num_total_iterations = 0
    j = 0

    agent.save_model(model_id)

    if f_use_testing_v2:
        print("Starting test v2")
    else:
        print("Starting test")

    start = time.perf_counter()

    max_num_winners = 0

    test_output_file.write("***************************************\n")

    for test_inputfile in test_filenames:
        test_profile = read_profile(test_inputfile)

        # print("Testing", test_inputfile)

        if f_use_testing_v2:
            winners = set(true_winners[j])
        else:
            winners = set()

        start_file = time.perf_counter()

        if f_use_testing_v2:
            test_PUT_winners, times_discovered, num_iters_to_find_all_winners = agent.test_model_v2(test_profile, winners)
        else:
            test_PUT_winners, times_discovered, num_iters_to_find_all_winners = agent.test_model(test_profile, num_test_iterations, winners)

        test_time = time.perf_counter() - start_file
        num_found_test += len(test_PUT_winners)
        num_total_iterations += num_iters_to_find_all_winners

        # print('Took', time.perf_counter() - start_file)
        # print(test_inputfile, test_PUT_winners, winners)
        missed_winners = winners - test_PUT_winners

        output_str = str(test_inputfile) + '\t' + str(list(test_PUT_winners)) + '\t' + str(len(test_PUT_winners)) + '\t' + str(missed_winners) + '\t' + str(len(missed_winners)) + '\t' + str(num_iters_to_find_all_winners) + '\t' + str(times_discovered) + '\t' + str(test_time)

        print(j, output_str)
        test_output_file.write(output_str + '\n')
        test_output_file.flush()
        max_num_winners = max(max_num_winners, len(test_PUT_winners))
        j += 1

    print("Test found", num_found_test, "in", num_total_iterations, "iterations, took", time.perf_counter() - start)
    # print("max num winners", max_num_winners)
    test_output_file.write("END" + "\t" + str(i) + "\t" + str(num_found_test) + "\t" + str(num_total_iterations) + "\n")
    test_output_file.flush()


# After how many profiles to test the model
test_every = 3500

# Whether or not to test before any training
test_at_start = 1

# note: can't use winners if shuffle
shuffle_training_data = 0

# Number of iterations to use when testing
num_test_iterations = 10

# Whether to initialize model from default values (for comparison purposes)
f_start_from_default = 0

f_experience_replay = 0

f_train_till_find_all_winners = 0
f_test_till_find_all_winners = 0

# v2 has network return values for all edges
f_use_v2 = 0

# testing v2 tests number of samples to find all winners
f_use_testing_v2 = 1

# Path to default model (used only if f_start_from_default)
# default_model_path = "C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\\\m10n10-100k\\default_agent_7_features.pth.tar"
default_model_path = "C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\RL\\results\\10-2\\results_RP_RL_main968971866_model.pth.tar"


if __name__ == '__main__':

    # Set random seeds
    random.seed(time.time())
    torch.manual_seed(time.time())

    start_everything = time.perf_counter()

    os.chdir(rpconfig.path)

    # Read the filenames
    # y1_filenames = rpconfig.filename
    # inf1 = open(y1_filenames, 'r')
    # prediction, filenames = read_Y_prediction(inf1)
    # inf1.close()

    # 14k m10n10
    filenames_file = open(rpconfig.filename_profiles, 'r')
    filenames = [i.strip('\n') for i in filenames_file]
    train_filenames = filenames[:10000] + filenames[11000:]
    test_filenames = filenames[10000:11000]   # the 1000 profiles we used in the paper
    # test_filenames = filenames[10000:10100]
    # test_filenames = ["M10N10-16804.csv"]

    # m10n10
    # filenames = sorted(glob.glob('M10N10-*.csv'))
    # train_filenames = filenames[0:80000]
    # test_filenames = filenames[80000:100000]

    # debugging
    # train_filenames = ['meh']
    # test_filenames = ['4circle.soc']

    # m50n50
    # filenames = sorted(glob.glob('M50N50-*.csv'))
    # train_filenames = filenames[0:1]
    # test_filenames = filenames[0:100]

    # Open files for output
    model_id = random.randint(0,1000000000)

    output_filename = "results_" + os.path.basename(__file__) + str(model_id)
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
    if f_use_v2:
        base = RL_base_v2(len(train_filenames))
    else:
        if f_experience_replay:
            base = RL_base_experience_replay(len(train_filenames))
        else:
            base = RL_base(len(train_filenames))

    # Create agent
    if f_use_v2:
        agent = RP_RL_agent_v2(base.learning_rate, loss_file)
    else:
        agent = RP_RL_agent(base.learning_rate, loss_file)

    if f_start_from_default:
        agent.load_model(default_model_path)

    num_profiles = 0
    total_time = 0

    total_winners_found = 0

    num_times_tested = 0

    # Counter variable
    i = 0

    print(output_filename)

    # Print header
    header = "Inputfile\tPUT-winners\tExploration Rate\tLearning Rate\tTau\tStop Conditions\tNum Nodes\tNum Winners Found\tLoss\tAvg Loss\tIs Acyclic\tIter To Find All Winners\tRuntime"
    print(header)
    output_file.write(header+'\n')

    # Print parameters
    parameters_file.write("Data Path\t" + rpconfig.path + '\n')
    parameters_file.write("Num Training Data\t" + str(len(train_filenames)) + '\n')
    parameters_file.write("Num Testing Data\t" + str(len(test_filenames)) + '\n')
    parameters_file.write("Train from...to\t" + train_filenames[0] + "\t" + train_filenames[-1] + '\n')
    parameters_file.write("Test from...to\t" + test_filenames[0] + "\t" + test_filenames[-1] + '\n')
    parameters_file.write("Training Data Shuffled\t" + str(shuffle_training_data) + '\n')
    parameters_file.write("Learning Rate\t" + str(base.learning_rate) + '\n')
    parameters_file.write("Learning Rate Decay\t" + str(base.f_learning_rate_decay) + '\n')
    if base.f_learning_rate_decay == 1:
        parameters_file.write("Learning Rate Decay Start\t" + str(base.learning_rate_start) + '\n')
        parameters_file.write("Learning Rate Decay End\t" + str(base.learning_rate_end) + '\n')
        parameters_file.write("Learning Rate Decay Rate\t" + str(base.learning_rate_decay) + '\n')
    parameters_file.write("Discount Factor\t" + str(base.discount_factor) + '\n')
    parameters_file.write("Exploration Rate\t" + str(base.exploration_rate) + '\n')
    parameters_file.write("Exploration Rate Decay\t" + str(base.f_exploration_rate_decay) + '\n')
    parameters_file.write("Exploration Type\t" + str(base.exploration_type) + '\n')
    if base.f_exploration_rate_decay:
        parameters_file.write("Exploration Rate Decay Start\t" + str(base.exploration_rate_start) + '\n')
        parameters_file.write("Exploration Rate Decay End\t" + str(base.exploration_rate_end) + '\n')
        parameters_file.write("Exploration Rate Decay Rate\t" + str(base.exploration_rate_decay) + '\n')
    parameters_file.write("Num Iterations per Profile\t" + str(base.num_iterations) + '\n')
    parameters_file.write("Agent D_in\t" + str(agent.D_in) + '\n')
    parameters_file.write("Agent H1\t" + str(agent.H1) + '\n')
    parameters_file.write("Agent H2\t" + str(agent.H2) + '\n')
    parameters_file.write("Agent D_out\t" + str(agent.D_out) + '\n')
    parameters_file.write("Num Polynomial Features\t" + str(agent.num_polynomial) + '\n')
    parameters_file.write("Use Visited Set\t" + str(agent.use_visited) + '\n')
    parameters_file.write("Use Cycles\t" + str(agent.use_cycles) + '\n')
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
    parameters_file.write("Update Target Network Every\t" + str(base.update_target_network_every) + '\n')
    parameters_file.write("Num Test Iterations\t" + str(num_test_iterations) + '\n')
    parameters_file.write("Train Till Find All Winners\t" + str(f_train_till_find_all_winners) + '\n')
    parameters_file.write("Test Till Find All Winners\t" + str(f_test_till_find_all_winners) + '\n')
    parameters_file.write("Use V2\t" + str(f_use_v2) + '\n')
    parameters_file.write("Shape Reward\t" + str(agent.f_shape_reward) + '\n')
    parameters_file.write("Use Testing V2\t" + str(f_use_testing_v2) + '\n')
    if f_use_testing_v2:
        parameters_file.write("Tau for Testing\t" + str(agent.tau_for_testing) + '\n')

    parameters_file.write("Date\t" + str(datetime.datetime.now()) + '\n')
    parameters_file.flush()

    if shuffle_training_data:
        random.shuffle(train_filenames)

    # Read true winners
    true_winners = []
    winners_file = open("../../Winners/winners_14k.txt", 'r')
    for line in winners_file:
        winners = []
        for c in line:
            if c == '[' or c == ',' or c == ' ' or c ==']' or c =='\n':
                continue
            winners.append(int(c))
        true_winners.append(winners)

    # split true_winners into train and test
    true_winners_train = true_winners[:10000] + true_winners[11000:]
    true_winners_test = true_winners[10000:11000]
    # true_winners_test = [[8, 1, 3, 6]]

    if not f_test_till_find_all_winners:
        test_output_file.write('Profile\tPUT-Winners\tNum Winners\tMissed Winners\tNum Missed Winners\tNum Iters\tIter Discoverd\tRuntime\n')
    else:
        test_header = "inputfile\tPUT-winners\tnum nodes\tdiscovery states\tmax discovery state\tdiscovery times\tmax discovery times\tstop condition hits\tsum stop cond hits\tnum hashes\tnum initial bridges\tnum redundant edges\ttime for cycles\truntime"
        test_output_file.write(test_header+'\n')

    for inputfile in train_filenames:
        if i % test_every == 0 and (test_at_start or i != 0):
            print("output:", output_filename)

            if f_test_till_find_all_winners:
                test_model_find_all_winners(test_output_file, agent, test_filenames, true_winners_test, model_id, num_times_tested)
            else:
                test_model(test_output_file, agent, test_filenames, true_winners_test, model_id)

            num_times_tested += 1

        if i % 500 == 0:
            agent.save_model(str(model_id) + "_" + str(i))

        profile = read_profile(inputfile)

        # Run the profile
        start = time.perf_counter()
        if f_train_till_find_all_winners:
            rp_results, iter_to_find_all_winners = base.reinforcement_loop(agent, profile, f_train_till_find_all_winners, set(true_winners_train[i]))
        else:
            rp_results, iter_to_find_all_winners = base.reinforcement_loop(agent, profile)
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

        result_text = "%s\t%r\t%f\t%f\t%f\t%r\t%d\t%d\t%f\t%f\t%s\t%d\t%f" % \
                      (inputfile, PUT_winners, base.exploration_rate, base.learning_rate, base.tau, stats.stop_condition_hits, stats.num_nodes, len(PUT_winners), stats.running_loss, avg_loss_per_node, is_acyclic, iter_to_find_all_winners, end - start)
        print(i, result_text)
        output_file.write(result_text + '\n')
        output_file.flush()

        # agent.print_model(weight_file, f_print_to_console=False)

        i += 1

    # Final test
    print("output:", output_filename)
    if f_test_till_find_all_winners:
        test_model_find_all_winners(test_output_file, agent, test_filenames, true_winners_test, model_id, num_times_tested)
    else:
        test_model(test_output_file, agent, test_filenames, true_winners, model_id)

    print("Total Time to Train: %f" % total_time)
    print("Average Time: %f" % (total_time / num_profiles))

    print("Total Runtime: %f" % (time.perf_counter() - start_everything))

    # Close files
    output_file.close()
    weight_file.close()
    test_output_file.close()
    parameters_file.close()
