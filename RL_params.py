# After how many profiles trained to test the model
test_every = 3500

# Whether or not to test before any training
test_at_start = 1

# Whether to shuffle the training data
# TODO: currently doesn't correspond with winners if shuffled
shuffle_training_data = 0

# Number of iterations to use when testing
# Doesn't matter if using test_till_find_all_winners of testing_v2
num_test_iterations = 10

# Whether to initialize model from default values (for comparison purposes)
f_start_from_default = 1

# Path to default model (used only if f_start_from_default)
default_model_path = "C:\\Users\shepht2\Documents\School\Masters\STV Ranked Pairs\\RL\\results\\10-3\\results_RP_RL_main161490381_model.pth.tar"

# Whether to use experience replay
f_experience_replay = 0

f_train_till_find_all_winners = 0

# Uses PUT_RP_using_model
f_test_till_find_all_winners = 0

# v2 has model return values for all edges
f_use_v2 = 0

# testing v2 tests number of samples to find all winners
f_use_testing_v2 = 1






def print_params(parameters_file):

