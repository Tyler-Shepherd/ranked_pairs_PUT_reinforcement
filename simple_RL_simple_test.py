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
from simple_RL_agent import simple_RL_agent

if __name__ == '__main__':

    maze = []
    for i in range(4):
        maze.append([])
        for j in range(4):
            maze[i].append(0)

    # i = random.randint(0, 4 - 1)
    # j = random.randint(0, 4 - 1)
    #
    # if (i != 0 or j != 0) and (i != 4 - 1 or j != 4 - 1):
    #     maze[i][j] = 1

    maze[0][1] = 1
    maze[1][1] = 1
    maze[3][1] = 1
    maze[3][2] = 1

    agent = simple_RL_agent()
    agent.load_model("checkpoint.pth.tar")
    agent.initialize(maze)
    agent.reset_environment()

    for row in maze:
        print(row)

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