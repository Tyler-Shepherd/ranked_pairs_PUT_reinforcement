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


# x by x grid
# different patterns of traps
# get to goal
# start at (0,0) -> top left
# goal is (size-1, size-1) -> bottom right
# features - whether action has trap, manhattan distance to goal
# can't wrap around
# should be able to make it through any path

# reward: 1 for goal, -1 for death


class simple_RL_stats():
    def __init__(self):
        self.num_turns = 0
        self.running_loss = 0


class simple_RL_agent():
    def __init__(self, learning_rate = 0, loss_output_file = None):
        # Initialize learning model

        self.D_in = 3  # input dimension, 6 features
        self.H = 2  # hidden dimension
        self.D_out = 1  # output dimension, just want q value

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.H, self.D_out)
        )

        self.loss_fn = torch.nn.MSELoss(size_average=False)  # using mean squared error

        self.running_loss = 0
        self.running_turns = 0
        self.print_loss_every = 1000

        self.loss_output_file = loss_output_file

        # 1 = gradient descent
        # 2 = adam
        self.optimizer_type = 1

        if self.optimizer_type == 2:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.stats = simple_RL_stats()

    '''
    Initializes environment for an iteration of learning
    env0 is a maze
    creates maze
    '''
    def initialize(self, env0):
        self.maze = env0
        self.maze_dimension = len(env0)

        self.stats = simple_RL_stats()



    '''
    Resets the row and col positions of agent to 0, 0
    '''
    def reset_environment(self):
        self.row = 0
        self.col = 0
        self.visited = set()

    '''
    Returns -1 if not at goal state
    Returns 1 if at bottom right corner
    Returns 2 if dead in trap
    '''
    def at_goal_state(self):
        if self.row == self.maze_dimension - 1 and self.col == self.maze_dimension - 1:
            return 1
        if self.maze[self.row][self.col] == 1:
            return 2
        return -1

    '''
    Returns (row, col) pairs if possible moves from current position
    '''
    def get_legal_actions(self):
        legal_actions = []

        if self.row - 1 >= 0 and (self.row-1, self.col) not in self.visited:
            legal_actions.append((self.row-1, self.col))
        if self.row + 1 < self.maze_dimension and (self.row+1, self.col) not in self.visited:
            legal_actions.append((self.row+1, self.col))
        if self.col - 1 >= 0 and (self.row, self.col-1) not in self.visited:
            legal_actions.append((self.row, self.col - 1))
        if self.col + 1 < self.maze_dimension and (self.row, self.col+1) not in self.visited:
            legal_actions.append((self.row, self.col + 1))

        return legal_actions


    def print_state(self):
        print("(row, col):", self.row, self.col)

    # Returns input layer features at current state taking action a
    # a is (row, col)
    def state_features(self, a):
        f = []
        f.append(2 * self.maze[a[0]][a[1]] - 1)
        if a[0] == a[1] == self.maze_dimension - 1:
            f.append(1)
        else:
            f.append(-1)
        f.append(abs(self.maze_dimension-1 - a[0]) + abs(self.maze_dimension-1 - a[1]))
        return Variable(torch.from_numpy(np.array(f)).float())

    def get_Q_val(self, a):
        state_features = self.state_features(a)
        return self.model(state_features)

    # Do nothing
    def goal_state_update(self):
        pass

    def print_model(self, output_file):
        for p in self.model.parameters():
            if output_file != "":
                data = p.data.numpy().tolist()
                for x in range(len(data)):
                    if isinstance(data[x],list) :
                        for j in range(len(data[x])):
                            output_file.write(str(data[x][j]) + " ")
                        output_file.write('\n')
                    else:
                        output_file.write(str(data[x]) + "\n")

            print(p)
        if output_file != "":
            output_file.write('\n')
            output_file.flush()

        print("Maze:")
        for row in self.maze:
            print(row)

        for r in range(self.maze_dimension):
            for c in range(self.maze_dimension):
                self.row = r
                self.col = c
                legal_actions = self.get_legal_actions()
                for a in legal_actions:
                    q_val = self.get_Q_val(a)
                    print("(" + str(r) + "," + str(c) + ")" + " to " + "(" + str(a[0]) + "," + str(a[1]) + "): " + str(q_val))


    '''
    Update agent position
    '''
    def make_move(self, a, f_testing = False):
        self.visited.add((self.row, self.col))

        self.row = a[0]
        self.col = a[1]

        if not f_testing:
            self.running_turns += 1

            if self.running_turns % self.print_loss_every == 0:
                print("*******LOSS:", self.running_loss)
                if self.loss_output_file:
                    self.loss_output_file.write(str(self.running_turns) + '\t' + str(self.running_loss) + '\n')
                    self.loss_output_file.flush()

                self.running_loss = 0



    def reward(self):
        current_state = self.at_goal_state()

        if current_state == -1:
            # Not a goal state
            reward_val =  0
        elif current_state == 2:
            # Death state
            reward_val = -1
        else:
            # Made it to goal
            reward_val = 1

        return torch.tensor(reward_val, dtype = torch.float32)

    def update_q(self, learning_rate, old_q_value, new_q_value):

        new_q_value = Variable(new_q_value)

        # Compute loss
        loss = self.loss_fn(old_q_value, new_q_value)

        self.stats.running_loss += loss.item()
        self.running_loss += loss.item()

        if self.optimizer_type == 1:
            # Gradient descent

            # Zero the gradients before running the backward pass.
            self.model.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()

            # Update the weights using gradient descent. Each parameter is a Tensor
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= learning_rate * param.grad
        elif self.optimizer_type == 2:
            # Adam

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()


    def save_model(self):
        # Save the model
        torch.save(self.model.state_dict(), "checkpoint.pth.tar")
        print("model saved")


    def test_model(self, test_env, num_iterations):
        self.initialize(test_env)

        # Sample using model greedily
        for iter in range(num_iterations):
            self.reset_environment()

            while self.at_goal_state() == -1:
                legal_actions = self.get_legal_actions()

                if len(legal_actions) == 0:
                    # No more edges can be added - goal state
                    break

                max_action = None
                max_action_val = float("-inf")
                for e in legal_actions:
                    action_val = self.get_Q_val(e)

                    if action_val > max_action_val:
                        max_action = e
                        max_action_val = action_val

                self.make_move(max_action, f_testing = True)


    def load_model(self, checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint)