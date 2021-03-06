import datetime

# Parameters not defined in here:
# In RP_RL_agent: Loss function
# In RP_RL_main: What training/test data to use for RL
# In main: Model architecture
# In RP_supervised_learning: SL loss function

run_SL = 0
run_RL = 1

# If 1, ignores model and uses LP heuristic from paper for q vals
test_with_LP = 0

# If 1, ignores model and selects actions randomly
test_with_random = 0

# If 1, runs 10x tests on test data at start
test_10x = 0

# PUT_agent learns how to fully search PUT-RP search space
f_use_PUT_agent = 0

############## Reinforcement Learning Parameters

# After how many profiles trained to test the model
# Runs test on validation data
test_every = 5000

# Whether or not to test before any RL training
test_at_start = 0

# Whether to shuffle the training data
shuffle_training_data = 0

# Number of iterations to use when testing
# Doesn't matter if using test_till_find_all_winners or testing_v2
num_test_iterations = 5

# Whether to use experience replay
f_experience_replay = 0

f_train_till_find_all_winners = 0

f_use_winners_distribution = 0

# Uses PUT_RP_using_model
f_test_using_PUT_RP = 0

# V2 has model return values over all potential edges, instead of just q value of single action
# reduces computation time, but harder to learn
f_use_v2 = 0

# Testing v2 tests number of samples to find all winners
f_use_testing_v2 = 1

learning_rate = 0.00001
# 0 = no decay
# 1 = decay over all profiles
# 2 = decay per profile (doesn't work)
# Note: If using Adam optimizer it doesn't matter
f_learning_rate_decay = 0
learning_rate_start = 0.9
learning_rate_end = 0.00001
learning_rate_decay = 2000000

exploration_rate = 0.4
f_exploration_rate_decay = 1
exploration_rate_start = 0.9
exploration_rate_end = 0.1
exploration_rate_decay = 600000

# used in boltzmann
tau_start = 1.0
tau_end = 0.05
tau_decay = 1000000

discount_factor = 0.95

# after how many iterations to update the target network to the agent's learned network
update_target_network_every = 1

# num iterations per profile
num_training_iterations = 1

# number of epochs, where one epoch = num_training_iterations per profile
num_epochs = 2



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
use_in_out = True # out/in of u,v
use_in_out_relative = False
use_total_degree = False
use_in_out_binary = False # binary out/in of u,v
use_in_out_matrix = False # in/out of every node
use_in_out_relative_matrix = False
use_total_degree_matrix = False # total degree of every node
use_in_out_binary_matrix = False # in/out binary of every node
use_K = True # u,v in K
use_K_big = False
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
use_betweenness_centrality = False


# Compute D_in from features used
D_in = 0
if use_in_out:
    D_in += 4
if use_in_out_relative:
    D_in += 4 * num_polynomial
if use_total_degree:
    D_in += num_polynomial * 2
if use_in_out_binary:
    D_in += 4
if use_in_out_matrix:
    D_in += 2 * m * num_polynomial
if use_in_out_relative_matrix:
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
if use_betweenness_centrality:
    D_in += 2 * num_polynomial
if f_use_PUT_agent:
    D_in += 1 # depth

D_in = int(D_in)

H1 = 10  # first hidden dimension
H2 = 32  # second hidden dimension

if f_use_v2:
    D_out = int(m * (m - 1)) # output dimension, values over all actions
else:
    D_out = 1 # just want q value


# After how many nodes to output summed loss
print_loss_every = 10000

# 1 = gradient descent
# 2 = adam
# 3 = stochastic gradient descent (pytorch optimizer)
optimizer_algo = 1

# Whether to shape reward by num times winner found
f_shape_reward = 0

# used in testing v2
tau_for_testing = 0.05
cutoff_testing_iterations = 1000

# if train_till_find_all_winners, stops after this many iterations
cutoff_training_iterations = 25000

# if f_use_PUT_agent, cutoff testing nodes in validation testing
cutoff_testing_nodes = 100000


# Experience Replay Parameters
train_from_experiences_every_iterations = 100

buffer_size = 5000
unusual_sample_factor = 0.9
batch_size = 500

# Supervised Learning Parameters

SL_test_every = 100
SL_test_at_start = 1

SL_optimal_action_learning_rate = 0.0001
SL_bad_action_learning_rate = 0.0001

SL_num_epochs = 200
SL_num_training_data = 1000
SL_num_test_data = 100







# debug_mode
# = 0: no output
# = 1: outputs minimal debug
# = 2: outputs general RL steps
# = 3: outputs everything
debug_mode = 0


# Print parameters
def print_params(parameters_file):
    parameters_file.write("run_SL\t" + str(run_SL) + '\n')
    parameters_file.write("run_RL\t" + str(run_RL) + '\n')

    parameters_file.write("test_with_LP\t" + str(test_with_LP) + '\n')
    parameters_file.write("test_with_random\t" + str(test_with_random) + '\n')
    parameters_file.write("test_10x\t" + str(test_10x) + '\n')

    parameters_file.write("f_use_PUT_agent\t" + str(f_use_PUT_agent) + '\n')

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
    parameters_file.write("num_epochs\t" + str(num_epochs) + '\n')
    parameters_file.write("Agent D_in\t" + str(D_in) + '\n')
    parameters_file.write("Agent H1\t" + str(H1) + '\n')
    parameters_file.write("Agent H2\t" + str(H2) + '\n')
    parameters_file.write("Agent D_out\t" + str(D_out) + '\n')
    parameters_file.write("Num Polynomial Features\t" + str(num_polynomial) + '\n')
    parameters_file.write("test_every\t" + str(test_every) + '\n')
    parameters_file.write("f_use_winners_distribution\t" + str(f_use_winners_distribution) + '\n')

    parameters_file.write("use_in_out\t" + str(use_in_out) + '\n')
    parameters_file.write("use_in_out_relative\t" + str(use_in_out_relative) + '\n')
    parameters_file.write("use_total_degree\t" + str(use_total_degree) + '\n')
    parameters_file.write("use_in_out_binary\t" + str(use_in_out_binary) + '\n')
    parameters_file.write("use_in_out_matrix\t" + str(use_in_out_matrix) + '\n')
    parameters_file.write("use_in_out_relative_matrix\t" + str(use_in_out_relative_matrix) + '\n')
    parameters_file.write("use_total_degree_matrix\t" + str(use_total_degree_matrix) + '\n')
    parameters_file.write("use_in_out_binary_matrix\t" + str(use_in_out_binary_matrix) + '\n')
    parameters_file.write("use_K\t" + str(use_K) + '\n')
    parameters_file.write("use_K_big\t" + str(use_K_big) + '\n')
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
    parameters_file.write("use_betweenness_centrality\t" + str(use_betweenness_centrality) + '\n')

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
