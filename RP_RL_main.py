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

from RL_base import RL_base
from RL_base_experience_replay import RL_base_experience_replay
from RP_RL_agent import RP_RL_agent
from RL_base_v2 import RL_base_v2
from RP_RL_agent_v2 import RP_RL_agent_v2
from RP_RL_agent_PUT import RP_RL_agent_PUT

import RP_utils

# Always call whatever parameter file you're using "params.py"
import params as params

# This file controls the main reinforcement learning
# Runs training and testing

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

'''
Runs PUT_RP algorithm to do full search of each profile, guaranteeing to find all PUT-winners
Uses agent model to evaluate local priority
'''
def test_model_using_PUT_RP(test_output_file, agent, test_filenames, model_id, num_times_tested):
    RP_utils.save_model(agent.model, "RL_test_" + str(num_times_tested), model_id)

    print("Starting PUT_RP_using_model test")
    test_output_file.write('***********************************\n')

    start_test = time.perf_counter()

    for test_inputfile in test_filenames:
        test_profile = read_profile(test_inputfile)

        start = time.perf_counter()
        if params.f_use_v2:
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

        print(result_text)
        test_output_file.write(result_text + '\n')
        test_output_file.flush()

    print("Time to test:", time.perf_counter() - start_test)

'''
Tests model using either v1 or v2, as set by params flag
test_output_file and test_output_summary_file are where results are printed to
agent has the current learned model
test_filenames is list of filenames to test on
true_winners is list of actual winners for each profile in test_filenames
model_id is identifying string
num_times_tested is int of how many times testing has been run
val_testing is whether or not this is for validation testing
'''
def test_model(test_output_file, test_output_summary_file, agent, test_filenames, true_winners, model_id, num_times_tested, val_testing):
    # Test the agents learned model
    if val_testing:
        RP_utils.save_model(agent.model, "RL_val_" + str(num_times_tested), model_id)
    else:
        RP_utils.save_model(agent.model, "RL_test_" + str(num_times_tested), model_id)

    num_found_test = 0
    num_total_iterations = 0
    time_to_test = 0
    total_100_discovery_nodes = 0
    total_100_discovery_runtime = 0
    j = 0

    print_string = "Starting"
    if val_testing:
        print_string += " validation testing"
    else:
        print_string += " test"
    if params.f_use_testing_v2:
        print_string += " v2"
    print_string += " " + str(num_times_tested)
    print(print_string)

    test_output_file.write("***************************************\n")

    for test_inputfile in test_filenames:
        test_profile = read_profile(test_inputfile)

        if params.f_use_testing_v2:
            winners = set(true_winners[j])
        else:
            winners = set()

        start_file = time.perf_counter()
        if params.f_use_testing_v2:
            test_PUT_winners, times_discovered, num_iters_to_find_all_winners, runtimes_discovered = agent.test_model_v2(test_profile, winners)
        else:
            test_PUT_winners, times_discovered, num_iters_to_find_all_winners, runtimes_discovered = agent.test_model(test_profile, start_file, val_testing)

        test_time = time.perf_counter() - start_file
        num_found_test += len(test_PUT_winners)
        num_total_iterations += num_iters_to_find_all_winners
        max_node_discovery = max(i for i in times_discovered.values())
        max_time_discovery = max(i for i in runtimes_discovered.values())

        total_100_discovery_nodes += max_node_discovery
        total_100_discovery_runtime += max_time_discovery

        time_to_test += test_time

        missed_winners = winners - test_PUT_winners

        output_str = str(test_inputfile) + \
                     '\t' + str(list(test_PUT_winners)) + \
                     '\t' + str(len(test_PUT_winners)) + \
                     '\t' + str(missed_winners) + \
                     '\t' + str(len(missed_winners)) + \
                     '\t' + str(num_iters_to_find_all_winners) + \
                     '\t' + str(times_discovered) + \
                     '\t' + str(max_node_discovery) + \
                     '\t' + str(runtimes_discovered) + \
                     '\t' + str(max_time_discovery) + \
                     '\t' + str(test_time)

        print(j, output_str)
        test_output_file.write(output_str + '\n')
        test_output_file.flush()
        j += 1

    print("Test found", num_found_test, "in", num_total_iterations, "iterations, took", time_to_test)
    test_output_summary_file.write(str(num_times_tested) + "\t" + str(num_found_test) + "\t" + str(num_total_iterations) + "\t" + str(num_total_iterations / len(test_filenames)) +
                                   "\t" + str(total_100_discovery_nodes / len(test_filenames)) + "\t" + str(time_to_test) + "\t" + str(time_to_test / len(test_filenames)) +
                                   "\t" + str(total_100_discovery_runtime / len(test_filenames)) + "\n")
    test_output_summary_file.flush()

    return num_total_iterations


class RP_RL():
    def RP_RL(self, model, model_id, parameters_file):
        start_RL = time.perf_counter()
        os.chdir(rpconfig.path)

        # 14k m10n10
        filenames_file = open(rpconfig.filename_profiles, 'r')
        filenames = [i.strip('\n') for i in filenames_file]
        train_filenames = filenames[:10000] + filenames[12000:]
        test_filenames = filenames[10000:11000]  # the same 1000 profiles we used in the paper
        validation_filenames = filenames[11000:11500]

        # m10n10
        # filenames = sorted(glob.glob('M10N10-*.csv'))
        # train_filenames = filenames[0:80000]
        # test_filenames = filenames[80000:100000]

        # m20n20
        # filenames_file = open(rpconfig.filename_profiles, 'r')
        # filenames = [i.strip('\n') for i in filenames_file]
        # train_filenames = filenames[:1] #+ filenames[12000:]
        # test_filenames = filenames[:1]
        # validation_filenames = filenames # all available m20n20

        # debugging
        # train_filenames = ['meh']
        # test_filenames = ['4circle.soc']

        # m50n50
        # filenames = sorted(glob.glob('M50N50-*.csv'))
        # train_filenames = filenames[0:1]
        # test_filenames = filenames[0:1000]

        # m40n40
        # filenames = sorted(glob.glob('M40N40-*.csv'))
        # train_filenames = filenames[0:1]
        # test_filenames = filenames[0:1000]

        # m50n50
        # filenames_file = open(rpconfig.filename_profiles, 'r')
        # filenames = [i.strip('\n') for i in filenames_file]
        # train_filenames = filenames[0:1]
        # test_filenames = filenames[0:1]
        # validation_filenames = filenames

        # Read true winners
        os.chdir(rpconfig.winners_path)
        true_winners = []
        winners_file = open("./winners_14k.txt", 'r')
        # winners_file = open("./winners_m20n20.txt", 'r')
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

        # Split true_winners into train and test
        true_winners_train = true_winners[:10000] + true_winners[12000:]
        true_winners_test = true_winners[10000:11000]
        true_winners_val = true_winners[11000:11100]

        # m20n20
        # true_winners_train = true_winners[:1]
        # true_winners_test = true_winners[:1]
        # true_winners_val = true_winners

        # Open files for output
        output_filename = str(model_id) + "_RL_training_results.txt"
        loss_filename = str(model_id) + "_RL_loss.txt"
        test_output_filename = str(model_id) + "_RL_test_results.txt"
        test_output_summary_filename = str(model_id) + "_RL_test_summary_results.txt"
        validation_output_filename = str(model_id) + "_RL_val_results.txt"
        validation_output_summary_filename = str(model_id) + "_RL_val_summary_results.txt"
        output_file = open(rpconfig.results_path + output_filename, "w+")
        test_output_file = open(rpconfig.results_path + test_output_filename, "w+")
        test_output_summary_file = open(rpconfig.results_path + test_output_summary_filename, "w+")
        val_output_file = open(rpconfig.results_path + validation_output_filename, "w+")
        val_output_summary_file = open(rpconfig.results_path + validation_output_summary_filename, "w+")
        loss_file = open(rpconfig.results_path + loss_filename, "w+")

        # Create RL base
        if params.f_use_v2:
            if params.f_experience_replay:
                print("Experience replay not implemented for v2")
                sys.exit(0)
            else:
                base = RL_base_v2(len(train_filenames))
        else:
            if params.f_use_PUT_agent and params.f_experience_replay:
                base = RL_base_PUT_agent_experience_replay(len(train_filenames))
            elif params.f_experience_replay:
                base = RL_base_experience_replay(len(train_filenames))
            else:
                base = RL_base(len(train_filenames))

        # Create agent
        if params.f_use_v2:
            agent = RP_RL_agent_v2(model, base.learning_rate, loss_file)
        else:
            if params.f_use_PUT_agent and params.f_experience_replay:
                agent = RP_RL_agent_PUT_experience(model, base.learning_rate, loss_file)
            elif params.f_use_PUT_agent:
                agent = RP_RL_agent_PUT(model, base.learning_rate, loss_file)
            else:
                agent = RP_RL_agent(model, base.learning_rate, loss_file)

        total_time = 0
        num_times_tested = 0

        print("***********************************************")
        print("Starting Reinforcement Learning", model_id)

        # Print header
        header = "Inputfile\tPUT-winners\tExploration Rate\tLearning Rate\tTau\tStop Conditions\tNum Nodes\tNum Winners Found\tLoss\tAvg Loss\tIs Acyclic\tIter To Find Winner\tIters To Find All Winners\tRunning Nodes\tWinners Dist\tnum_iters_reset_skipped\tNum hashed\tRuntime"
        print(header)
        output_file.write(header + '\n')
        output_file.flush()

        loss_file.write('Num Nodes' + '\t' + 'Loss Per Node' + '\n')
        loss_file.flush()

        # Open winner distribution file
        if params.f_use_winners_distribution:
            winners_distribution_file = open(rpconfig.winners_distribution_filename, 'r')
            winners_distribution = {}
            for line in winners_distribution_file:
                line = line.strip('\n')
                line = line.split('\t')
                if len(line) == 1:
                    current_file = line[0]
                    continue
                if current_file not in winners_distribution:
                    winners_distribution[current_file] = {}
                winners_distribution[current_file][int(line[0])] = int(line[1])
            winners_distribution_file.close()

        # Shuffle training data
        if params.shuffle_training_data:
            combined = list(zip(train_filenames, true_winners_train))
            random.shuffle(combined)
            train_filenames, true_winners_train = zip(*combined)

        # Print test output file heading
        if params.f_test_using_PUT_RP:
            test_header = "inputfile\tPUT-winners\tnum nodes\tdiscovery states\tmax discovery state\tdiscovery times\tmax discovery times\tstop condition hits\tsum stop cond hits\tnum hashes\tnum initial bridges\tnum redundant edges\ttime for cycles\truntime"
        elif params.f_use_PUT_agent:
            test_header = 'Profile\tPUT-Winners\tNum Winners\tMissed Winners\tNum Missed Winners\tNum Nodes\tNode Discovered\t100% Nodes\tRuntime Discovered\t100% Runtime\tRuntime'
            test_summary_header = "Test\tNum PUT-Winners Found\tTotal Num Nodes\tAvg Nodes Per Profile\tAvg 100% Nodes\tTotal Time\tAvg Time Per Profile\tAvg 100% Time\n"
            test_output_summary_file.write(test_summary_header)
            val_output_summary_file.write(test_summary_header)
            test_output_summary_file.flush()
            val_output_summary_file.flush()
        else:
            test_header = 'Profile\tPUT-Winners\tNum Winners\tMissed Winners\tNum Missed Winners\tNum Iters\tIter Discoverd\tMax Iter Discovery\tTime Discovered\tMax Time Discovery\tRuntime'
            test_summary_header = "Test\tNum PUT-Winners Found\tTotal Num Iterations\tAvg Iterations Per Profile\tAvg 100% Iters\tTotal Time\tAvg Time Per Profile\tAvg 100% Time\n"
            test_output_summary_file.write(test_summary_header)
            val_output_summary_file.write(test_summary_header)
            test_output_summary_file.flush()
            val_output_summary_file.flush()
        test_output_file.write(test_header + '\n')
        val_output_file.write(test_header + '\n')
        test_output_file.flush()
        val_output_file.flush()

        # Print additional parameters
        parameters_file.write("RL Data Path\t" + rpconfig.path + '\n')
        parameters_file.write("RL Num Training Data\t" + str(len(train_filenames)) + '\n')
        parameters_file.write("RL Num Testing Data\t" + str(len(test_filenames)) + '\n')
        parameters_file.write("RL Train From...To\t" + train_filenames[0] + "\t" + train_filenames[-1] + '\n')
        parameters_file.write("RL Test From...To\t" + test_filenames[0] + "\t" + test_filenames[-1] + '\n')
        parameters_file.write("RL Loss Function\t" + str(agent.loss_fn) + '\n')
        parameters_file.flush()

        val_results = []

        if params.test_10x:
            print("********** testing 10x *******************")
            assert params.f_start_from_default or params.test_with_LP
            for t in range(10):
                test_model(test_output_file, test_output_summary_file, agent, test_filenames, true_winners_test, model_id, "final_" + str(t), False)

        # assert not params.f_use_testing_v2
        # assert params.f_start_from_default
        # num_samples_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # runtimes
        # for num_samples in num_samples_range:
        #     params.num_test_iterations = num_samples
        #     print(params.num_test_iterations)
        #     start = time.perf_counter()
        #     test_model(val_output_file, val_output_summary_file, agent, validation_filenames, true_winners_val, model_id, num_times_tested, True)
        #     num_times_tested += 1
        # return

        for epoch in range(params.num_epochs):
            i = 0
            print('---------------Epoch ' + str(epoch) + '------------------------')

            # Shuffle training data
            if params.shuffle_training_data:
                combined = list(zip(train_filenames, true_winners_train))
                random.shuffle(combined)
                train_filenames, true_winners_train = zip(*combined)

            for inputfile in train_filenames:
                # Test model on validation data
                # Not necessary since epochs added
                # if i % params.test_every == 0 and (params.test_at_start or i != 0):
                #     if params.f_test_using_PUT_RP:
                #         test_model_using_PUT_RP(test_output_file, agent, test_filenames, model_id, num_times_tested)
                #     else:
                #         num_iters = test_model(val_output_file, val_output_summary_file, agent, validation_filenames, true_winners_val, model_id, num_times_tested, True)
                #         val_results.append(num_iters)
                #
                #     num_times_tested += 1

                if i % 10 == 0:
                    RP_utils.save_model(model, "RL_" + str(i), model_id)

                profile = read_profile(inputfile)

                # Run the profile
                print(inputfile)
                start = time.perf_counter()
                if params.f_use_winners_distribution:
                    rp_results, iter_to_find_winner, iter_to_find_all_winners = base.reinforcement_loop(agent, profile, winners_distribution = winners_distribution[inputfile])
                elif params.f_train_till_find_all_winners:
                    rp_results, iter_to_find_winner, iter_to_find_all_winners = base.reinforcement_loop(agent, profile, true_winners = set(true_winners_train[i]))
                else:
                    rp_results, iter_to_find_winner, iter_to_find_all_winners = base.reinforcement_loop(agent, profile, true_winners = set(true_winners_train[i]), filename = inputfile)
                end = time.perf_counter()

                # Evaluate and output results
                PUT_winners = sorted(rp_results.known_winners)
                stats = agent.stats

                total_time += (end - start)

                if stats.num_nodes == 0:
                    avg_loss_per_node = 0
                else:
                    avg_loss_per_node = stats.running_loss / stats.num_nodes

                is_acyclic = str(nx.is_directed_acyclic_graph(agent.E_0))

                if params.f_use_winners_distribution:
                    output_winners_distribution = winners_distribution[inputfile]
                else:
                    output_winners_distribution = {}

                result_text = "%s\t%r\t%f\t%f\t%f\t%r\t%d\t%d\t%f\t%f\t%s\t%r\t%d\t%d\t%r\t%d\t%d\t%f" % \
                              (inputfile, PUT_winners, base.exploration_rate, base.learning_rate, base.tau,
                               stats.stop_condition_hits, stats.num_nodes, len(PUT_winners), stats.running_loss,
                               avg_loss_per_node, is_acyclic, iter_to_find_winner, iter_to_find_all_winners, agent.running_nodes, output_winners_distribution, stats.num_iters_reset_skipped, agent.stats.num_hashed, end - start)
                print(i, result_text)
                output_file.write(result_text + '\n')
                output_file.flush()

                i += 1

            # Test on validation data after each epoch
            if params.f_test_using_PUT_RP:
                test_model_using_PUT_RP(test_output_file, agent, test_filenames, model_id, num_times_tested)
            else:
                num_iters = test_model(val_output_file, val_output_summary_file, agent, validation_filenames, true_winners_val, model_id, num_times_tested, True)
                val_results.append(num_iters)
            num_times_tested += 1

        print('----------------------Training Done------------------------------')
        print("Validation results:", val_results)
        best_model = np.argmin(val_results)
        print("Best model:", best_model)

        # Use best model from validation testing to test 10x on test set
        RP_utils.load_model(model, rpconfig.results_path + str(model_id) + "_RL_val_" + str(best_model) + "_model.pth.tar")
        # Create agent
        if params.f_use_v2:
            agent_testing = RP_RL_agent_v2(model, base.learning_rate)
        else:
            agent_testing = RP_RL_agent(model, base.learning_rate)

        for t in range(10):
            test_model(test_output_file, test_output_summary_file, agent_testing, test_filenames, true_winners_test, model_id, "final_" + str(t), False)

        print("Total Time to Train: %f" % total_time)
        print("Average Time Per Profile: %f" % (total_time / len(train_filenames)))

        print("Total RL Runtime: %f" % (time.perf_counter() - start_RL))

        # Close files
        output_file.close()
        test_output_file.close()
        test_output_summary_file.close()
        val_output_file.close()
        val_output_summary_file.close()
        loss_file.close()