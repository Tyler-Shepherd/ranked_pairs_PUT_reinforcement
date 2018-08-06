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
from simple_RL_agent import simple_RL_agent

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

        test_PUT_winners = agent.test_model(test_profile, num_test_iterations)

        num_found_test += len(test_PUT_winners)

        j += 1

    print("Test found", num_found_test, "took", time.perf_counter() - start)
    test_output_file.write(str(i) + "\t" + str(num_found_test) + "\n")
    test_output_file.flush()


# After how many mazes to test the model
test_every = 2500

# Whether or not to test before any training
test_at_start = 0

# Number of iterations to use when testing
num_test_iterations = 10

# Whether to initialize model from default values (for comparison purposes)
f_start_from_default = 0

# Path to default model (used only if f_start_from_default)
default_model_path = "C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\\\m10n10-100k\\default_agent.pth.tar"


if __name__ == '__main__':

    # Set random seeds
    random.seed(time.time())
    torch.manual_seed(time.time())

    maze_dimension = 4
    num_mazes = 1000

    # row by column
    mazes = []

    for m in range(num_mazes):
        new_maze = []
        for i in range(maze_dimension):
            new_maze.append([])
            for j in range(maze_dimension):
                new_maze[i].append(0)

        i = random.randint(0, maze_dimension - 1)
        j = random.randint(0, maze_dimension - 1)

        if (i != 0 or j != 0) and (i != maze_dimension - 1 or j != maze_dimension - 1):
            new_maze[i][j] = 1

        mazes.append(new_maze)

    # for maze in mazes:
    #     for row in maze:
    #         print(row)
    #     print('\n')

    train_mazes = mazes[:1000]
    test_mazes = []

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

    # Create agent and base
    base = RL_base(len(train_mazes))
    agent = simple_RL_agent(base.learning_rate, loss_file)

    if f_start_from_default:
        agent.load_model(default_model_path)

    num_mazes = 0
    total_time = 0

    # Counter variable
    i = 0

    # Print header
    # header = "Inputfile\tPUT-winners\tStop Conditions\tNum Nodes\tNum Winners Found\tLoss\tAvg Loss\tRuntime"
    # print(header)
    # output_file.write(header+'\n')

    # Print parameters
    # parameters_file.write("Data Path\t" + rpconfig.path + '\n')
    # parameters_file.write("Num Training Data\t" + str(len(train_filenames)) + '\n')
    # parameters_file.write("Num Testing Data\t" + str(len(test_filenames)) + '\n')
    # parameters_file.write("Train from...to\t" + train_filenames[0] + "\t" + train_filenames[-1] + '\n')
    # parameters_file.write("Test from...to\t" + test_filenames[0] + "\t" + test_filenames[-1] + '\n')
    # parameters_file.write("Learning Rate\t" + str(base.learning_rate) + '\n')
    # parameters_file.write("Learning Rate Decay\t" + str(base.f_learning_rate_decay) + '\n')
    # parameters_file.write("Discount Factor\t" + str(base.discount_factor) + '\n')
    # parameters_file.write("Exploration Rate\t" + str(base.exploration_rate) + '\n')
    # parameters_file.write("Exploration Rate Decay\t" + str(base.f_exploration_rate_decay) + '\n')
    # parameters_file.write("Num Iterations per Profile\t" + str(base.num_iterations) + '\n')
    # parameters_file.write("Agent D_in\t" + str(agent.D_in) + '\n')
    # parameters_file.write("Agent H\t" + str(agent.H) + '\n')
    # parameters_file.write("Agent D_out\t" + str(agent.D_out) + '\n')
    # parameters_file.write("Agent Model\t" + str(agent.model) + '\n')
    # parameters_file.write("Agent Loss Function\t" + str(agent.loss_fn) + '\n')
    # parameters_file.write("Agent Optimizer Type\t" + str(agent.optimizer_type) + '\n')
    # if f_start_from_default:
    #     parameters_file.write("Default Model File\t" + default_model_path + '\n')
    # else:
    #     parameters_file.write("Default Model File\tN/A\n")
    # parameters_file.write("Date\t" + str(datetime.datetime.now()) + '\n')
    #
    # parameters_file.flush()

    for inputmaze in train_mazes:
        if i % test_every == 0 and (test_at_start or i != 0):
            test_model(test_output_file, agent, test_mazes)

        # Run the maze
        start = time.perf_counter()
        rp_results = base.reinforcement_loop(agent, inputmaze)
        end = time.perf_counter()

        stats = agent.stats

        total_time += (end - start)
        num_mazes += 1

        if stats.num_turns == 0:
            avg_loss_per_turn = 0
        else:
            avg_loss_per_turn = stats.running_loss / stats.num_turns

        result_text = "%d\t%f\t%f\t%f" % (stats.num_turns, stats.running_loss, avg_loss_per_turn, end - start)
        print(i, result_text)
        output_file.write(result_text + '\n')
        output_file.flush()

        agent.print_model(weight_file)
        agent.save_model()

        i += 1

    # Final test
    # test_model(test_output_file, agent, test_mazes)

    # Close files
    output_file.close()
    weight_file.close()
    test_output_file.close()
    parameters_file.close()

    print("Total Time to Train: %f" % total_time)
    print("Average Time: %f" % (total_time / num_mazes))