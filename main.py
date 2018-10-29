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
from networkx.algorithms.connectivity import local_edge_connectivity
from networkx.algorithms.connectivity import local_node_connectivity
import rpconfig
from collections import defaultdict
import matplotlib.pyplot as plt
from queue import PriorityQueue
import torch
from torch.autograd import Variable
import random
from pprint import pprint
import glob

import params as params
from RP_supervised_learning import RP_SL
from RP_supervised_learning_v2 import RP_SL_v2
from RP_RL_main import RP_RL
import RP_utils

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == '__main__':
    # Set random seeds
    random.seed(time.time())
    torch.manual_seed(time.time())

    # Create model
    if params.f_use_v2:
        model = torch.nn.Sequential(
            torch.nn.Linear(params.D_in, params.H1),
            torch.nn.ReLU(),
            torch.nn.Linear(params.H1, params.H2),
            torch.nn.ReLU(),
            torch.nn.Linear(params.H2, params.H2),
            torch.nn.ReLU(),
            torch.nn.Linear(params.H2, params.D_out),
            torch.nn.Softmax(dim=0)
        )
        if params.use_in_out or params.use_total_degree or params.use_in_out_binary or params.use_K or params.use_voting_rules or params.use_cycles or params.use_connectivity:
            print("Features not implemented for v2")
            sys.exit(0)
        if params.optimizer_algo == 3:
            print("Stochastic gradient descent not implemented for v2")
            sys.exit(0)
        if params.f_use_winners_distribution:
            print("Winners distribution not done for v2")
            sys.exit(0)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(params.D_in, params.D_out)
        )
        if params.use_in_out_matrix or params.use_total_degree_matrix or params.use_in_out_binary_matrix or params.use_voting_rules_matrix or params.use_connectivity_matrix:
            print("Features not implemented for v1")
            sys.exit(0)
        if params.optimizer_algo == 3 and params.f_learning_rate_decay != 0:
            print("Decaying learning rate with stochastic gradient descent not implemented")
            sys.exit(0)

    if params.f_experience_replay and (params.use_visited or params.use_cycles):
        print('Features not implemented for experience replay')
        sys.exit(0)

    if not params.f_use_testing_v2:
        print('No other testing but v2 implemented with validation testing')
        sys.exit(0)

    print("Model:")
    print(model)

    model.apply(init_weights)

    # Identifying id for this run
    model_id = random.randint(0, 1000000000)

    if params.f_start_from_default:
        RP_utils.load_model(model, params.default_model_path)

    RP_utils.save_model(model, "init", model_id)

    # Print parameters
    parameters_output_filename = rpconfig.results_path + str(model_id) + "_parameters.txt"
    parameters_file = open(parameters_output_filename, "w+")
    params.print_params(parameters_file)
    parameters_file.write("Agent Model\t" + str(model) + '\n')

    # Run SL (based on param flag)
    if params.run_SL:
        if params.f_use_v2:
            RP_SL_v2().RP_SL_v2(model, model_id, parameters_file)
        else:
            RP_SL().RP_SL(model, model_id, parameters_file)

    # Run RL (based on param flag)
    if params.run_RL:
        RP_RL().RP_RL(model, model_id, parameters_file)

    parameters_file.close()