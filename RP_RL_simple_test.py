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

    agent.K = frozenset({0})

    while agent.at_goal_state() == -1:

        legal_actions = agent.get_legal_actions()

        # Find best action
        max_action = None
        max_action_val = float("-inf")
        for e in legal_actions:
            action_Q_val = agent.get_Q_val(e)
            print(e, action_Q_val)
            if action_Q_val > max_action_val:
                max_action = e
                max_action_val = action_Q_val

        print("Max action", max_action)

        agent.make_move(max_action)