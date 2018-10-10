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
#from RL_base_experience_replay import RL_base_experience_replay
from RP_RL_agent import RP_RL_agent
#from RP_RL_agent_node2vec import RP_RL_agent_node2vec
from RL_base_v2 import RL_base_v2
from RP_RL_agent_v2 import RP_RL_agent_v2

import RP_utils

# Always call whatever parameter file you're using "params.py"
import params as params

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
Tests model
'''
def test_model(test_output_file, agent, test_filenames, true_winners, model_id, num_times_tested):
    # Test the agents learned model
    RP_utils.save_model(agent.model, "RL_test_" + str(num_times_tested), model_id)

    num_found_test = 0
    num_total_iterations = 0
    j = 0

    if params.f_use_testing_v2:
        print("Starting test v2", num_times_tested)
    else:
        print("Starting test", num_times_tested)
    test_output_file.write("***************************************\n")

    start_test = time.perf_counter()

    for test_inputfile in test_filenames:
        test_profile = read_profile(test_inputfile)

        start_file = time.perf_counter()

        winners = set(true_winners[j])

        if params.f_use_testing_v2:
            test_PUT_winners, times_discovered, num_iters_to_find_all_winners = agent.test_model_v2(test_profile, winners)
        else:
            test_PUT_winners, times_discovered, num_iters_to_find_all_winners = agent.test_model(test_profile)

        test_time = time.perf_counter() - start_file
        num_found_test += len(test_PUT_winners)
        num_total_iterations += num_iters_to_find_all_winners

        missed_winners = winners - test_PUT_winners

        output_str = str(test_inputfile) + '\t' + str(list(test_PUT_winners)) + '\t' + str(len(test_PUT_winners)) + '\t' + str(missed_winners) + '\t' + str(len(missed_winners)) + '\t' + str(num_iters_to_find_all_winners) + '\t' + str(times_discovered) + '\t' + str(test_time)

        print(j, output_str)
        test_output_file.write(output_str + '\n')
        test_output_file.flush()
        j += 1

    print("Test found", num_found_test, "in", num_total_iterations, "iterations, took", time.perf_counter() - start_test)
    test_output_file.write("END" + "\t" + str(num_times_tested) + "\t" + str(num_found_test) + "\t" + str(num_total_iterations) + "\n")
    test_output_file.flush()


class RP_RL():
    def RP_RL(self, model, model_id, parameters_file):
        start_RL = time.perf_counter()
        os.chdir(rpconfig.path)

        # 14k m10n10
        filenames_file = open(rpconfig.filename_profiles, 'r')
        filenames = [i.strip('\n') for i in filenames_file]
        train_filenames = filenames[:10000] + filenames[11000:]
        test_filenames = filenames[10000:11000]  # the 1000 profiles we used in the paper

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
        output_filename = str(model_id) + "_RL_training_results.txt"
        loss_filename = str(model_id) + "_RL_loss.txt"
        test_output_filename = str(model_id) + "_RL_test_results.txt"
        output_file = open(rpconfig.results_path + output_filename, "w+")
        test_output_file = open(rpconfig.results_path + test_output_filename, "w+")
        loss_file = open(rpconfig.results_path + loss_filename, "w+")

        # Create RL base
        if params.f_use_v2:
            if params.f_experience_replay:
                print("Experience replay not implemented for v2")
                sys.exit(0)
            else:
                base = RL_base_v2(len(train_filenames))
        else:
            if params.f_experience_replay:
                base = RL_base_experience_replay(len(train_filenames))
            else:
                base = RL_base(len(train_filenames))

        # Create agent
        if params.f_use_v2:
            agent = RP_RL_agent_v2(model, base.learning_rate, loss_file)
        else:
            agent = RP_RL_agent(model, base.learning_rate, loss_file)

        # Counter variable
        i = 0

        total_time = 0
        num_times_tested = 0

        print("***********************************************")
        print("Starting Reinforcement Learning", model_id)

        # Print header
        header = "Inputfile\tPUT-winners\tExploration Rate\tLearning Rate\tTau\tStop Conditions\tNum Nodes\tNum Winners Found\tLoss\tAvg Loss\tIs Acyclic\tIter To Find Winner\tIters To Find All Winners\tRuntime"
        print(header)
        output_file.write(header + '\n')
        output_file.flush()

        loss_file.write('Num Nodes' + '\t' + 'Loss Per Node' + '\n')
        loss_file.flush()

        # Read true winners
        os.chdir(rpconfig.winners_path)
        true_winners = []
        winners_file = open("./winners_14k.txt", 'r')
        for line in winners_file:
            winners = []
            for c in line:
                if c == '[' or c == ',' or c == ' ' or c == ']' or c == '\n':
                    continue
                winners.append(int(c))
            true_winners.append(winners)
        os.chdir(rpconfig.path)

        # Split true_winners into train and test
        true_winners_train = true_winners[:10000] + true_winners[11000:]
        true_winners_test = true_winners[10000:11000]

        # Shuffle training data
        if params.shuffle_training_data:
            combined = list(zip(train_filenames, true_winners_train))
            random.shuffle(combined)
            train_filenames, true_winners_train = zip(*combined)

        # Print test output file heading
        if not params.f_test_using_PUT_RP:
            test_header = 'Profile\tPUT-Winners\tNum Winners\tMissed Winners\tNum Missed Winners\tNum Iters\tIter Discoverd\tRuntime'
        else:
            test_header = "inputfile\tPUT-winners\tnum nodes\tdiscovery states\tmax discovery state\tdiscovery times\tmax discovery times\tstop condition hits\tsum stop cond hits\tnum hashes\tnum initial bridges\tnum redundant edges\ttime for cycles\truntime"
        test_output_file.write(test_header + '\n')

        # Print additional parameters
        parameters_file.write("RL Data Path\t" + rpconfig.path + '\n')
        parameters_file.write("RL Num Training Data\t" + str(len(train_filenames)) + '\n')
        parameters_file.write("RL Num Testing Data\t" + str(len(test_filenames)) + '\n')
        parameters_file.write("RL Train From...To\t" + train_filenames[0] + "\t" + train_filenames[-1] + '\n')
        parameters_file.write("RL Test From...To\t" + test_filenames[0] + "\t" + test_filenames[-1] + '\n')
        parameters_file.write("RL Loss Function\t" + str(agent.loss_fn) + '\n')
        parameters_file.flush()

        for inputfile in train_filenames:
            # Test model
            if i % params.test_every == 0 and (params.test_at_start or i != 0):
                if params.f_test_using_PUT_RP:
                    test_model_using_PUT_RP(test_output_file, agent, test_filenames, model_id, num_times_tested)
                else:
                    test_model(test_output_file, agent, test_filenames, true_winners_test, model_id, num_times_tested)

                num_times_tested += 1

            if i % 500 == 0:
                RP_utils.save_model(model, "RL_" + str(i), model_id)

            profile = read_profile(inputfile)

            # Run the profile
            print(inputfile)
            start = time.perf_counter()
            if params.f_train_till_find_all_winners:
                rp_results, iter_to_find_winner, iter_to_find_all_winners = base.reinforcement_loop(agent, profile, True, set(true_winners_train[i]))
            else:
                rp_results, iter_to_find_winner, iter_to_find_all_winners = base.reinforcement_loop(agent, profile)
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

            result_text = "%s\t%r\t%f\t%f\t%f\t%r\t%d\t%d\t%f\t%f\t%s\t%r\t%d\t%f" % \
                          (inputfile, PUT_winners, base.exploration_rate, base.learning_rate, base.tau,
                           stats.stop_condition_hits, stats.num_nodes, len(PUT_winners), stats.running_loss,
                           avg_loss_per_node, is_acyclic, iter_to_find_winner, iter_to_find_all_winners, end - start)
            print(i, result_text)
            output_file.write(result_text + '\n')
            output_file.flush()

            i += 1

        # Final test
        if params.f_test_using_PUT_RP:
            test_model_using_PUT_RP(test_output_file, agent, test_filenames, model_id, num_times_tested)
        else:
            test_model(test_output_file, agent, test_filenames, true_winners, model_id, num_times_tested)

        print("Total Time to Train: %f" % total_time)
        print("Average Time Per Profile: %f" % (total_time / len(train_filenames)))

        print("Total RL Runtime: %f" % (time.perf_counter() - start_RL))

        # Close files
        output_file.close()
        test_output_file.close()
        loss_file.close()