import datetime

# Parameters not defined in here:
# In RP_RL_agent: Loss function
# In RP_RL_main: What training/test data to use for RL
# In main: Model architecture
# In RP_supervised_learning: SL loss function

run_SL = 0
run_RL = 1

############## Reinforcement Learning Parameters

# After how many profiles trained to test the model
test_every = 10000000

# Whether or not to test before any RL training
test_at_start = 0

# Whether to shuffle the training data
shuffle_training_data = 0

# Number of iterations to use when testing
# Doesn't matter if using test_till_find_all_winners or testing_v2
num_test_iterations = 10

# Whether to use experience replay
f_experience_replay = 0

f_train_till_find_all_winners = 0

f_use_winners_distribution = 0

# Uses PUT_RP_using_model
f_test_using_PUT_RP = 0

# V2 has model return values for all edges
f_use_v2 = 0

# Testing v2 tests number of samples to find all winners
f_use_testing_v2 = 1

learning_rate = 0.05
# 0 = no decay
# 1 = decay over all profiles
# 2 = decay per profile (doesn't work)
# Note: If using Adam optimizer it doesn't matter
f_learning_rate_decay = 1
learning_rate_start = 0.8
learning_rate_end = 0.01
learning_rate_decay = 2000000

exploration_rate = 0.4
f_exploration_rate_decay = 1
exploration_rate_start = 0.9
exploration_rate_end = 0.1
exploration_rate_decay = 600000

# used in boltzmann
tau_start = 1.0
tau_end = 0.05
tau_decay = 1500000

discount_factor = 0.95

# after how many iterations to update the target network to the agent's learned network
update_target_network_every = 25

num_training_iterations = 200

# 1 = eps greedy
# 2 = boltzmann
exploration_algo = 2

# Num candidates
m = 10.0
# Num voters
n = 10.0


################## Model Parameters

# Whether to initialize model from default values (for comparison purposes)
f_start_from_default = 0

# Path to default model (used only if f_start_from_default)
default_model_path = "C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\RL\\results\\10-3\\results_RP_RL_main161490381_model.pth.tar"

# What features to include
num_polynomial = 1
use_in_out = False # out/in of u,v
use_total_degree = False
use_in_out_binary = False # binary out/in of u,v
use_in_out_matrix = False # in/out of every node
use_total_degree_matrix = False # total degree of every node
use_in_out_binary_matrix = False # in/out binary of every node
use_K = True # u,v in K
use_voting_rules = False
use_voting_rules_matrix = False
use_edge_weight = False
use_visited = False # I don't think visited makes sense as a feature when using SL
use_cycles = False
use_vectorized_wmg = False
use_posmat = False
use_tier_adjacency_matrix = False # adjacency matrix of just the legal actions
use_connectivity = False # takes forever to compute, don't use it
use_connectivity_matrix = False # takes forever to compute, don't use it

use_adjacency_matrix = False
use_K_representation = False


# Compute D_in from features used
D_in = 0
if use_in_out:
    D_in += num_polynomial * 4
if use_total_degree:
    D_in += num_polynomial * 2
if use_in_out_binary:
    D_in += 4
if use_in_out_matrix:
    D_in += 2 * m * num_polynomial
if use_total_degree_matrix:
    D_in += m * num_polynomial
if use_in_out_binary_matrix:
    D_in += 2 * m
if use_K:
    D_in += 2
if use_voting_rules:
    D_in += num_polynomial * 8
if use_voting_rules_matrix:
    D_in += m * 4 * num_polynomial
if use_edge_weight:
    D_in += num_polynomial
if use_visited:
    D_in += num_polynomial
if use_cycles:
    D_in += num_polynomial
if use_vectorized_wmg:
    D_in += m * (m-1)
if use_posmat:
    D_in += m * m
if use_adjacency_matrix:
    D_in += m * m
if use_tier_adjacency_matrix:
    D_in += m * m
if use_connectivity:
    D_in += num_polynomial * 4
if use_connectivity_matrix:
    D_in += m * (m - 1) * 2 * num_polynomial
if use_K_representation:
    D_in += m

D_in = int(D_in)

H1 = 1000  # first hidden dimension
H2 = 1000  # second hidden dimension

if f_use_v2:
    D_out = int(m * (m - 1)) # output dimension, values over all actions
else:
    D_out = 1 # just want q value


# After how many nodes to output summed loss
print_loss_every = 1000

# 1 = gradient descent
# 2 = adam
# 3 = stochastic gradient descent (pytorch optimizer)
optimizer_algo = 1

# Whether to shape reward by num times winner found
f_shape_reward = 0

# used in testing v2
tau_for_testing = 0.1
cutoff_testing_iterations = 1000

# if train_till_find_all_winners, stops after this many iterations
cutoff_training_iterations = 25000


# Experience Replay Parameters
train_from_experiences_every_iterations = 100

buffer_size = 5000
unusual_sample_factor = 0.9
batch_size = 500

# Supervised Learning Parameters

SL_test_every = 100
SL_test_at_start = 1

SL_optimal_action_learning_rate = 0.05
SL_bad_action_learning_rate = 0.005

SL_num_epochs = 500
SL_num_training_data = 1000
SL_num_test_data = 1000







# debug_mode
# = 0: no output
# = 1: outputs minimal debug
# = 2: outputs general RL steps
# = 3: outputs everything
debug_mode = 0


# Print parameters
def print_params(parameters_file):
    parameters_file.write("Training Data Shuffled\t" + str(shuffle_training_data) + '\n')
    parameters_file.write("Learning Rate Decay\t" + str(f_learning_rate_decay) + '\n')
    if f_learning_rate_decay == 0:
        parameters_file.write("Learning Rate\t" + str(learning_rate) + '\n')
    elif f_learning_rate_decay == 1:
        parameters_file.write("Learning Rate Start\t" + str(learning_rate_start) + '\n')
        parameters_file.write("Learning Rate End\t" + str(learning_rate_end) + '\n')
        parameters_file.write("Learning Rate Decay Rate\t" + str(learning_rate_decay) + '\n')
    parameters_file.write("Discount Factor\t" + str(discount_factor) + '\n')
    parameters_file.write("Exploration Rate Decay\t" + str(f_exploration_rate_decay) + '\n')
    if not f_exploration_rate_decay:
        parameters_file.write("Exploration Rate\t" + str(exploration_rate) + '\n')
    else:
        parameters_file.write("Exploration Rate Decay Start\t" + str(exploration_rate_start) + '\n')
        parameters_file.write("Exploration Rate Decay End\t" + str(exploration_rate_end) + '\n')
        parameters_file.write("Exploration Rate Decay Rate\t" + str(exploration_rate_decay) + '\n')
    parameters_file.write("Exploration Algo\t" + str(exploration_algo) + '\n')
    if exploration_algo == 2:
        parameters_file.write("Tau Start\t" + str(tau_start) + '\n')
        parameters_file.write("Tau End\t" + str(tau_end) + '\n')
        parameters_file.write("Tau Decay\t" + str(tau_decay) + '\n')
    parameters_file.write("Num Training Iterations per Profile\t" + str(num_training_iterations) + '\n')
    parameters_file.write("Agent D_in\t" + str(D_in) + '\n')
    parameters_file.write("Agent H1\t" + str(H1) + '\n')
    parameters_file.write("Agent H2\t" + str(H2) + '\n')
    parameters_file.write("Agent D_out\t" + str(D_out) + '\n')
    parameters_file.write("Num Polynomial Features\t" + str(num_polynomial) + '\n')
    parameters_file.write("test_every\t" + str(test_every) + '\n')

    parameters_file.write("use_in_out\t" + str(use_in_out) + '\n')
    parameters_file.write("use_total_degree\t" + str(use_total_degree) + '\n')
    parameters_file.write("use_in_out_binary\t" + str(use_in_out_binary) + '\n')
    parameters_file.write("use_in_out_matrix\t" + str(use_in_out_matrix) + '\n')
    parameters_file.write("use_total_degree_matrix\t" + str(use_total_degree_matrix) + '\n')
    parameters_file.write("use_in_out_binary_matrix\t" + str(use_in_out_binary_matrix) + '\n')
    parameters_file.write("use_K\t" + str(use_K) + '\n')
    parameters_file.write("use_voting_rules\t" + str(use_voting_rules) + '\n')
    parameters_file.write("use_voting_rules_matrix\t" + str(use_voting_rules_matrix) + '\n')
    parameters_file.write("use_edge_weight\t" + str(use_edge_weight) + '\n')
    parameters_file.write("use_visited\t" + str(use_visited) + '\n')
    parameters_file.write("use_cycles\t" + str(use_cycles) + '\n')
    parameters_file.write("use_vectorized_wmg\t" + str(use_vectorized_wmg) + '\n')
    parameters_file.write("use_posmat\t" + str(use_posmat) + '\n')
    parameters_file.write("use_tier_adjacency_matrix\t" + str(use_tier_adjacency_matrix) + '\n')
    parameters_file.write("use_connectivity\t" + str(use_connectivity) + '\n')
    parameters_file.write("use_connectivity_matrix\t" + str(use_connectivity_matrix) + '\n')
    parameters_file.write("use_adjacency_matrix\t" + str(use_adjacency_matrix) + '\n')
    parameters_file.write("use_K_representation\t" + str(use_K_representation) + '\n')

    parameters_file.write("Agent Optimizer Algo\t" + str(optimizer_algo) + '\n')
    if f_start_from_default:
        parameters_file.write("Default Model File\t" + default_model_path + '\n')
    else:
        parameters_file.write("Default Model File\tN/A\n")
    parameters_file.write("Experience Replay\t" + str(f_experience_replay) + '\n')
    if f_experience_replay:
        parameters_file.write("Experience Replay Buffer Size\t" + str(buffer_size) + '\n')
        parameters_file.write("Experience Replay Sample Factor\t" + str(unusual_sample_factor) + '\n')
        parameters_file.write("Experience Replay Batch Size\t" + str(batch_size) + '\n')
        parameters_file.write("Experience Replay Train Every\t" + str(train_from_experiences_every_iterations) + '\n')
    else:
        parameters_file.write("print_loss_every\t" + str(print_loss_every) + '\n')
    parameters_file.write("Update Target Network Every\t" + str(update_target_network_every) + '\n')
    if not f_test_using_PUT_RP and not f_use_testing_v2:
        parameters_file.write("Num Test Iterations\t" + str(num_test_iterations) + '\n')
    parameters_file.write("Train Till Find All Winners\t" + str(f_train_till_find_all_winners) + '\n')
    if f_train_till_find_all_winners:
        parameters_file.write("cutoff_training_iterations\t" + str(cutoff_training_iterations) + '\n')
    parameters_file.write("Test Using PUT_RP\t" + str(f_test_using_PUT_RP) + '\n')
    parameters_file.write("Use V2\t" + str(f_use_v2) + '\n')
    parameters_file.write("Shape Reward\t" + str(f_shape_reward) + '\n')
    parameters_file.write("Use Testing V2\t" + str(f_use_testing_v2) + '\n')
    parameters_file.write("Tau for Testing\t" + str(tau_for_testing) + '\n')
    if f_use_testing_v2:
        parameters_file.write("cutoff_testing_iterations\t" + str(cutoff_testing_iterations) + '\n')

    parameters_file.write("SL test every\t" + str(SL_test_every) + '\n')
    parameters_file.write("SL_optimal_action_learning_rate\t" + str(SL_optimal_action_learning_rate) + '\n')
    parameters_file.write("SL_bad_action_learning_rate\t" + str(SL_bad_action_learning_rate) + '\n')
    parameters_file.write("SL_num_epochs\t" + str(SL_num_epochs) + '\n')
    parameters_file.write("SL_num_training_data\t" + str(SL_num_training_data) + '\n')
    parameters_file.write("SL_num_test_data\t" + str(SL_num_test_data) + '\n')

    parameters_file.write("Date\t" + str(datetime.datetime.now()) + '\n')

    parameters_file.flush()
