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


def save_model(model, details_str, model_id):
    # Save the model
    model_checkpoint  = str(model_id) + "_" + details_str + "_model.pth.tar"
    torch.save(model.state_dict(), rpconfig.results_path + model_checkpoint)
    print("model saved to", model_checkpoint)

def load_model(model, checkpoint_filename):
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint)
    print("loaded model from", checkpoint_filename)