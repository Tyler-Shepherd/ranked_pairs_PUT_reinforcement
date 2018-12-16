# ranked_pairs_PUT_reinforcement
Applying reinforcement learning to learn efficient heuristics for the PUT (parallel universes tiebreaking) problem for Tideman's ranked pairs social choice algorithm.

Ranked pairs PUT algorithm based off "Practical Algorithms for STV and Ranked Pairs with Parallel Universes Tiebreaking" by Jun Wang, Sujoy Sikdar, Tyler Shepherd, Zhibing Zhao, Chunheng Jiang and Lirong Xia from AAAI-19.

main.py runs the code, params.py has configurable parameters.

rpconfig.py defines paths to relevant needed data files.

PUT_RP_sampling.py can be run independently - it uses the learned model to sample RP a fixed number of times.
  Set params.default_model_path and params.f_start_from_default
  Set params.num_testing_iterations for the number of samples to run


Run DAgger/RP_DAgger_DFS.py for DAgger algorithm based learning.
