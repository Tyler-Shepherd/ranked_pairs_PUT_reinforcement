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

import params
import RP_utils

if __name__ == '__main__':

    os.chdir("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\soc-toc-jw\soc-toc-jw\\")
    # os.chdir("C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\data\\m10n10-100k")

    # profile = read_profile("4circle.soc")
    profile = read_profile("10circle.soc")
    # profile = read_profile("M10N10-100000.csv")

    model = torch.nn.Sequential(
        torch.nn.Linear(params.D_in, params.H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(params.H1, params.H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(params.H2, params.D_out)
    )

    RP_utils.load_model(model, "C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\RL\\results\\10-3\\results_RP_RL_main161490381_model.pth.tar")

    RP_utils.print_feature_weights(model)

    agent = RP_RL_agent(model)
    # agent = RP_RL_agent_v2(model)

    agent.initialize(profile)
    agent.reset_environment()

    agent.K = frozenset({0, 1, 2, 4, 5, 6, 8, 9})

    # print("********* START *************")
    #
    # while agent.at_goal_state()[0] == -1:
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


