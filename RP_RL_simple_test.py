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

if __name__ == '__main__':

    os.chdir("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\soc-toc-jw\soc-toc-jw\\")

    profile = read_profile("4circle.soc")

    agent = RP_RL_agent()
    agent.load_model("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\\\m10n10-100k\\checkpoint.pth.tar")
    agent.initialize(profile)
    agent.reset_environment()

    output_file = open("weight_output.txt", 'w+')

    agent.print_model(output_file)

    agent.K = frozenset({0,1,2})

    print("********* START *************")

    # Print weights
    model = agent.model.parameters()

    is_bias = -1
    to_hidden = -1

    bias_weight_sum_to_hidden = 0
    feature_weight_sum_to_hidden = [0] * agent.D_in

    for p in model:
        data = p.data.numpy().tolist()

        if is_bias == -1:
            for unit_weights in data:
                for i in range(len(unit_weights)):
                    if to_hidden == -1:
                        feature_weight_sum_to_hidden[i] += unit_weights[i]
        else:
            if to_hidden == -1:
                for val in data:
                    bias_weight_sum_to_hidden += val

        if is_bias == 1:
            to_hidden = 1
        is_bias = - is_bias

    # print(feature_weight_sum_to_hidden)

    print("bias", 1)
    print("u out degree", feature_weight_sum_to_hidden[0] / bias_weight_sum_to_hidden)
    print("u in degree", feature_weight_sum_to_hidden[1]/ bias_weight_sum_to_hidden)
    print("v out degree", feature_weight_sum_to_hidden[2] / bias_weight_sum_to_hidden)
    print("v in degree", feature_weight_sum_to_hidden[3] / bias_weight_sum_to_hidden)
    print("u known winner", feature_weight_sum_to_hidden[4] / bias_weight_sum_to_hidden)
    print("v known winner", feature_weight_sum_to_hidden[5] / bias_weight_sum_to_hidden)
    print("edge cycles num", feature_weight_sum_to_hidden[6] / bias_weight_sum_to_hidden)
    print("num times visited", feature_weight_sum_to_hidden[7] / bias_weight_sum_to_hidden)

    # for i in feature_weight_sum_to_hidden:
    #     print(i / bias_weight_sum_to_hidden)

    # while agent.at_goal_state() == -1:
    #
    #     legal_actions = agent.get_legal_actions()
    #
    #     # Find best action
    #     max_action = None
    #     max_action_val = float("-inf")
    #     print("\nlegal actions:")
    #     for e in legal_actions:
    #         action_Q_val = agent.get_Q_val(e)
    #         print(e, action_Q_val)
    #         if action_Q_val > max_action_val:
    #             max_action = e
    #             max_action_val = action_Q_val
    #
    #     print("Max action", max_action)
    #
    #     agent.make_move(max_action)