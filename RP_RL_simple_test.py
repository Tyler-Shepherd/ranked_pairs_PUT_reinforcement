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

sys.path.append('./RL')
from RP_RL_main import read_profile
from RP_RL_agent import RP_RL_agent
from RP_RL_agent_v2 import RP_RL_agent_v2

def print_weights(agent):
    # Print weights
    model = agent.model.parameters()

    is_bias = -1
    to_hidden = -1

    bias_weight_to_hidden = []
    # first index is feature unit, second index is hidden unit
    feature_weight_to_hidden = []
    for i in range(agent.D_in):
        new_list = []
        for j in range(agent.H):
            new_list.append(0)
        feature_weight_to_hidden.append(new_list)
    hidden_weight_to_output = [0] * agent.H

    for p in model:
        data = p.data.numpy().tolist()

        if is_bias == -1:
            for j in range(len(data)):
                unit_weights = data[j]
                # unit_weights has all the weights to a hidden unit
                for i in range(len(unit_weights)):
                    if to_hidden == -1:
                        feature_weight_to_hidden[i][j] = unit_weights[i]
                        # feature_weight_sum_to_hidden[i] += unit_weights[i]
                    else:
                        hidden_weight_to_output[i] += unit_weights[i]
        else:
            if to_hidden == -1:
                for val in data:
                    bias_weight_to_hidden.append(val)
            else:
                bias_weight_to_output = data[0]

        if is_bias == 1:
            to_hidden = 1
        is_bias = - is_bias

    bias_weight_sum = sum(bias_weight_to_hidden[i] * hidden_weight_to_output[i] for i in range(agent.H))

    feature_weight_sum = [0] * agent.D_in

    for i in range(agent.D_in):
        for j in range(agent.H):
            feature_weight_sum[i] += feature_weight_to_hidden[i][j] * hidden_weight_to_output[j]

    # print("bias", 1)
    # print("u out degree", feature_weight_sum_to_hidden[0] / bias_weight_sum)
    # print("u in degree", feature_weight_sum_to_hidden[1]/ bias_weight_sum)
    # print("v out degree", feature_weight_sum_to_hidden[2] / bias_weight_sum)
    # print("v in degree", feature_weight_sum_to_hidden[3] / bias_weight_sum)
    # print("u known winner", feature_weight_sum_to_hidden[4] / bias_weight_sum)
    # print("v known winner", feature_weight_sum_to_hidden[5] / bias_weight_sum)
    # print("edge cycles num", feature_weight_sum_to_hidden[6] / bias_weight_sum)
    # print("num times visited", feature_weight_sum_to_hidden[7] / bias_weight_sum)

    print("bias input\t", bias_weight_sum)
    print("bias hidden\t", bias_weight_to_output)
    for i in feature_weight_sum:
        print(i / bias_weight_sum)

if __name__ == '__main__':

    os.chdir("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\soc-toc-jw\soc-toc-jw\\")
    # os.chdir("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\m10n10-100k")

    # profile = read_profile("4circle.soc")
    profile = read_profile("10circle.soc")
    # profile = read_profile("M10N10-100000.csv")

    agent = RP_RL_agent()
    # agent = RP_RL_agent_v2()

    # agent.load_model("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\\\m10n10-100k\\checkpoint.pth.tar")
    agent.load_model("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\RL\\results\\10-1\\results_RP_RL_main882861866_model.pth.tar")

    agent.initialize(profile)
    agent.reset_environment()

    output_file = open("weight_output.txt", 'w+')
    agent.print_model(output_file)

    agent.K = frozenset({0, 1, 2, 4, 5, 6, 8, 9})

    print("********* START *************")

    # print_weights(agent)

    while agent.at_goal_state()[0] == -1:

        legal_actions = agent.get_legal_actions()

        # Find best action
        max_action = None
        max_action_val = float("-inf")
        print("\nlegal actions:")
        for e in legal_actions:
            action_Q_val = agent.get_Q_val(e)
            print(e, action_Q_val)
            if action_Q_val > max_action_val:
                max_action = e
                max_action_val = action_Q_val

        print("Max action", max_action)

        agent.make_move(max_action)


    # v2

    # while agent.at_goal_state() == -1:
    #
    #     legal_actions = agent.get_legal_actions()
    #
    #     state_q_vals = agent.get_Q_vals()
    #
    #     # Find best action
    #     max_action = None
    #     max_action_val = float("-inf")
    #     print("\nlegal actions:")
    #     for e in legal_actions:
    #         action_Q_val = state_q_vals[e]
    #         print(e, action_Q_val)
    #         if action_Q_val > max_action_val:
    #             max_action = e
    #             max_action_val = action_Q_val
    #
    #     print("Max action", max_action)
    #
    #     agent.make_move(max_action)


