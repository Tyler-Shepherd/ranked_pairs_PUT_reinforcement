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
    model = torch.nn.Sequential(
        torch.nn.Linear(params.D_in, params.H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(params.H1, params.H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(params.H2, params.D_out)
    )
    model.apply(init_weights)
    # TODO: include softmax if v2

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
        RP_SL().RP_SL(model, model_id)

    # Run RL (based on param flag)
    if params.run_RL:
        RP_RL().RP_RL(model, model_id, parameters_file)

    parameters_file.close()