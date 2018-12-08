import io
import os
import prefpy_io
import math
import time
from numpy import *
import numpy as np
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
import networkx.algorithms.connectivity
import glob
import torch
from torch.autograd import Variable
from pprint import pprint

# Copied from LPwinners_db_5_6
# Does not use database


# computes the plurality scores of candidates given an input profile
# input: profile of preferences as np matrix
# output: m-vector of plurality scores of candidates, normalized by n
def plurality_score(profile_matrix):
    (n,m) = np.shape(profile_matrix)
    pluralityscores = [0] * m
    for i in range(n):
        pluralityscores[profile_matrix[i,0]] += 1
    pluralityscores_normalized = list(1.*np.array(pluralityscores)/n)
    return pluralityscores_normalized

#computes the Borda scores of candidates given an input profile
# input: profile
# output: m-vector of Borda scores of candidates, normalized by n(m-1)
def borda_score(profile_matrix):
    (n,m) = np.shape(profile_matrix)
    bordascores = [0] * m
    for i in range(n):
        for j in range(m):
            bordascores[profile_matrix[i,j]] += (m - j)
    bordascores_normalized = list(1.*np.array(bordascores)/(n*(m-1)))
    return bordascores_normalized

#computes the Copeland scores of candidates
# input: wmg dict
# output: m-vector of Copeland scores of candidates, normalized by m-1 to [-1, 1]
def copeland_score(wmg):
    m = len(wmg.keys())
    copelandscores = [0] * m
    for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
        if wmg[cand1][cand2] > 0:
            copelandscores[cand1] += 1
            copelandscores[cand2] -= 1
    copelandscores_normalized = list(1.*np.array(copelandscores)/(m-1))
    return copelandscores_normalized

#computes the Maximin scores of candidates
# input: wmg dict
# output: m-vector of Maximin scores of candidates, normalized by n to [-1, 1]
def maximin_score(wmg):
    n = len(wmg.keys())
    maximinscores = [0] * n
    for cand in wmg.keys():
        maximinscores[cand] = min(i for (_, i) in wmg[cand].items())

    maximinscores_normalized = list(1.*np.array(maximinscores)/n)
    return maximinscores_normalized

# just vectorizes the wmg
# input: wmg
# output: vectorized weighted majority graph. sorted by candidates, then by opponents,
#   normalized by no. of voters
def vectorize_wmg(wmg):
    m = len(wmg)
    n = np.sum(np.abs([wmg[0][i] for i in range(1,m)]))
    wmg_vec = [wmg[i][j] for i in range(m) for j in range(m) if not j == i]
    wmg_vec_normalized = list(1.*np.array(wmg_vec)/n)
    return wmg_vec_normalized


# creates a positional matrix and vectorizes it
# input: profile
# intermediate: positional matrix posmat
#   posmat[i][j] = # voters ranking candidate i in position j
# output: vectorized positional matrix, sorted by candidate, then by position,
#   normalized by no. of voters
def profile2posmat(profile_matrix):
    (n,m) = np.shape(profile_matrix)
    posmat = np.zeros((m,m))

    for i in range(n):
        vote = profile_matrix[i, :]
        for pos in range(m):
            cand = vote[0, pos]
            posmat[cand][pos] += 1
    posmat_vec = posmat.flatten()
    posmat_vec_normalized = list(1.*np.array(posmat_vec)/n)
    return posmat_vec_normalized

# For node s, avg over all other nodes t of local edge connectivity = num edges needed to remove to disconnect s and t
def avg_edge_connectivity(G, I, s):
    total_connectivity = 0
    for t in I:
        if t != s:
            total_connectivity += local_edge_connectivity(G, s, t)

    avg_connectivity = total_connectivity / (len(I) - 1)
    # TODO: normalize
    return avg_connectivity

# For node s, avg over all other nodes t of local node connectivity = num nodes needed to remove to disconnect s and t
def avg_node_connectivity(G, I, s):
    total_connectivity = 0
    for t in I:
        if t != s:
            total_connectivity += local_node_connectivity(G, s, t)

    avg_connectivity = total_connectivity / (len(I) - 1)
    # TODO: normalize
    return avg_connectivity




class PUT_RP_using_policy():
    """
    The Ranked Pairs mechanism.
    """

    # debug_mode
    # = 0: no output
    # = 1: outputs only initial state
    # = 2: outputs on stop conditions
    # = 3: outputs all data
    def __init__(self):
        global debug_mode
        self.debug_mode = 0

        # Timeout in seconds
        self.TIMEOUT = 60 * 60 * 60

    class Stats:
        # Stores statistics being measured and updated throughout procedure
        """
        Stopping Conditions:
            1: G U E is acyclic
            2: possible_winners <= known_winners (pruning)
            3: exactly 1 cand with in degree 0
            4: G U Tier is acyclic (in max children method)
        """
        def __init__(self):
            self.discovery_states = dict()
            self.discovery_times = dict()
            self.num_nodes = 0
            self.stop_condition_hits = {1: 0, 2: 0, 3: 0, 4: 0}
            self.num_hashes = 0
            self.num_initial_bridges = 0
            self.num_redundant_edges = 0
            self.time_for_cycles = 0

            # contains list of tuples ((G,W), e) of state and edge selected
            self.strategy = []

    def output_graph(self, G):
        # Draws the given graph G using networkx

        pos = nx.circular_layout(G)  # positions for all nodes
        pos = dict(zip(sorted(pos.keys()), pos.values()))
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=350)

        # edges
        nx.draw_networkx_edges(G, pos, width=3, alpha=0.5, edge_color='b')

        # labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')

        plt.axis('off')
        plt.savefig("weighted_graph.png")  # save as png
        plt.show()  # display

    def add_winners(self, G, I, known_winners, stats, possible_winners = None):
        """
        Adds the winners of completed RP graph G
        :param G: networkx graph, should be final resulting graph after running RP
        :param I: list of all nodes
        :param known_winners: list of winners found so far, will be updated
        :param stats: Stats class storing run statistics
        :param possible_winners: Can optionally pass in possible winners if already computed to avoid re-computing here
        """
        if possible_winners is None:
            G_in_degree = G.in_degree(I)
            to_be_added = set([x[0] for x in G_in_degree if x[1] == 0])
        else:
            to_be_added = possible_winners
        for c in to_be_added:
            if c not in known_winners:
                known_winners.add(c)
                stats.discovery_states[c] = stats.num_nodes
                stats.discovery_times[c] = time.perf_counter() - self.BEGIN

                if self.debug_mode >= 1:
                    print("Found new winner:", c)
                    print(G.edges())

    def stop_conditions(self, G, E, T, I, known_winners, stats):
        """
        Determines if G, E state can be ended early
        :param G: networkx DiGraph of the current representation of "locked in" edges in RP
        :param E: networkx DiGraph of the remaining edges not yet considered
        :param I: list of all nodes
        :param known_winners: list of currently known PUT-winners
        :param stats: Stats object containing runtime statistics
        :return: -1 if no stop condition met, otherwise returns the int of the stop condition
        """

        in_deg = G.in_degree(I)
        possible_winners = [x[0] for x in in_deg if x[1] == 0]

        # Stop Condition 2: Pruning. Possible winners are subset of known winners
        if set(possible_winners) <= known_winners:
            stats.stop_condition_hits[2] += 1
            if self.debug_mode >= 3:
                print("Stop Condition 2: pruned")
            return 2

        # Stop Condition 3: Exactly one node has indegree 0
        if len(possible_winners) == 1:
            stats.stop_condition_hits[3] += 1
            if self.debug_mode >= 2:
                print("Stop Condition 3: one cand in degree 0")
            self.add_winners(G, I, known_winners, stats, possible_winners)
            return 3

        # Stop Condition 1: G U E U T is acyclic
        # temp_G = nx.compose(G, E)
        # temp_G.add_edges_from(T)
        # if nx.is_directed_acyclic_graph(temp_G) is True:
        #     stats.stop_condition_hits[1] += 1
        #     if self.debug_mode >= 2:
        #         print("Stop Condition 1: acyclic")
        #     self.add_winners(G, I, known_winners, stats)
        #     return 1

        return -1


    # Returns array of val to each power from 1 to i
    def polynomialize(self, val, i):
        return [val**j for j in range(1, i+1)]

    # Returns input layer features at current state taking action a
    # a is an edge
    def state_features(self, G, I, K):
        f = []

        # in/out matrices
        out_degree = G.out_degree(I)
        for (i, j) in out_degree:
            f.append(j / len(I))

        in_degree = G.in_degree(I)
        for (i, j) in in_degree:
            f.append(j / len(I))

        # K representation
        K_list = []
        for i in I:
            if i in K:
                K_list.append(1)
            else:
                K_list.append(0)
        f.extend(K_list)

        if self.debug_mode >= 3:
            print("features", f)

        return Variable(torch.from_numpy(np.array(f)).float())



    def sample_strategy(self, profile, policy):
        """
        Returns 1. a list of all PUT-winners of profile under ranked pairs rule
        and 2. A Stats object of runtime statistics

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        self.BEGIN = time.perf_counter()

        # Initialize
        stats = self.Stats()

        wmg = profile.getWmg()
        known_winners = set()
        I = list(wmg.keys())

        # save profile as matrix for use in features
        # profile_matrix = []
        # for p in profile.preferences:
        #     profile_matrix.append(p.getOrderVector())
        # self.profile_matrix = np.asmatrix(profile_matrix)

        G = nx.DiGraph()
        G.add_nodes_from(I)

        E = nx.DiGraph()
        E.add_nodes_from(I)
        self.max_edge_weight = 0
        for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
            if wmg[cand1][cand2] > 0:
                E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])
                self.max_edge_weight = max(self.max_edge_weight, wmg[cand1][cand2])

        # self.E_0 = E.copy()
        # self.adjacency_0 = nx.adjacency_matrix(E, nodelist=I).todense()

        # compute voting rules scores
        # self.plurality_scores = plurality_score(self.profile_matrix)
        # self.borda_scores = borda_score(self.profile_matrix)
        # self.copeland_scores = copeland_score(wmg)
        # self.maximin_scores = maximin_score(wmg)
        #
        # self.vectorized_wmg = vectorize_wmg(wmg)
        # self.posmat = profile2posmat(self.profile_matrix)

        # print(wmg)
        # self.output_graph(E)

        # Add any bridge edges from any tier in E
        # These are guaranteed to never be in a cycle, so will always be in the final graph after RP procedure
        Gc = G.copy()
        Gc.add_edges_from(E.edges())
        scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if len(g.edges()) != 0]
        bridges = set(Gc.edges()) - set(itertools.chain(*scc))
        G.add_edges_from(bridges)
        E.remove_edges_from(bridges)

        stats.num_initial_bridges = len(bridges)

        # order the edges for model ease
        edges_ordered = {}
        index = 0
        for i in range(len(I)):
            for j in range(len(I)):
                if i != j:
                    edges_ordered[(i,j)] = index
                    index += 1

        # Each node contains (G, E, tier)
        # root = Node(value=(self.edges2string(G.edges(), I), self.edges2string(E.edges(), I)))
        root = Node(value=(G, E, [], None))
        stackNode = []
        stackNode.append(root)

        hashtable = set()

        END = self.BEGIN + self.TIMEOUT

        while stackNode:
            # Pop new node to explore
            node = stackNode.pop()
            (G, E, T, e) = node.value

            if time.perf_counter() > END:
                print("TIMEOUT")
                return stats.strategy, stats.num_nodes, True

            # Check hash
            hash_state = self.edges2string(G.edges(), I) + self.edges2string(E.edges(), I) + self.edges2string(T, I)
            if hash_state in hashtable:
                stats.num_hashes += 1
                if self.debug_mode == 3:
                    print("hashed in outer hashtable")
                continue
            hashtable.add(hash_state)

            stats.num_nodes += 1
            if e is not None:
                Gc = G.copy()
                Gc.remove_edges_from([e])
                stats.strategy.append(((self.edges2string(Gc.edges(), I), frozenset(known_winners)), e))

            if self.debug_mode == 3:
                print("Popped new node: ")
                print("G:", sorted(G.edges()))
                print("E:", sorted(E.edges()))
                print("T:", sorted(T))

            # Flag for whether expanding the current tier required finding max children
            f_found_max_children = 0

            # Continue performing RP on this state as long as tie-breaking order doesn't matter
            while len(E.edges()) != 0 or len(T) != 0:
                if self.stop_conditions(G, E, T, I, known_winners, stats) != -1:
                    # Stop condition hit
                    break

                if len(T) == 0:
                    # Get a new tier
                    (max_weight, max_edge) = max([(d['weight'], (u, v)) for (u, v, d) in E.edges(data=True)])
                    T = [(u, v) for (u, v, d) in E.edges(data=True) if d['weight'] == max_weight]
                    E.remove_edges_from(T)

                    if self.debug_mode == 3:
                        print("New tier =", T)

                    if len(T) == 1:
                        # Tier only has one edge, just add it
                        if self.debug_mode == 3:
                            print("Only 1 edge in tier")

                        if nx.has_path(G, max_edge[1], max_edge[0]) is False:
                            stats.strategy.append(((self.edges2string(G.edges(), I), frozenset(known_winners)), max_edge))
                            G.add_edges_from(T)
                        continue

                # Perform reductions every step:

                # Compute "bridge edges" which are not in any cycle
                Gc = G.copy()
                Gc.add_edges_from(T)
                scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if len(g.edges()) != 0]
                bridges = set(Gc.edges()) - set(itertools.chain(*scc))
                G.add_edges_from(bridges)
                T = list(set(T) - bridges)

                G_tc = nx.transitive_closure(G)

                # Remove "inconsistent edges" that cannot be added to G without causing cycle
                reverse_G = nx.DiGraph.reverse(G_tc)
                T = list(set(T) - set(reverse_G.edges()))

                # Remove "redundant edges": if there is already path from e[0] to e[1], can immediately add e
                redundant_edges = set()
                for e in T:
                    if G_tc.has_edge(e[0], e[1]):
                        redundant_edges.add(e)
                        G.add_edges_from([e])
                stats.num_redundant_edges += len(redundant_edges)
                T = list(set(T) - redundant_edges)

                if len(T) == 0:
                    # No need to find further children, as tier is now empty
                    if self.debug_mode == 3:
                        print("Tier empty")
                    continue

                # Add each edge to stack by priority
                children = []
                children_probs = []
                T = sorted(T)
                EUT = E.copy()
                EUT.add_edges_from(T)

                probs = policy(self.state_features(G, I, known_winners))

                for e in T:
                    if not G_tc.has_edge(e[1], e[0]):
                        f_found_max_children = 1

                        Gc = G.copy()
                        Gc.add_edges_from([e])
                        Ec = E.copy()
                        Tc = copy.deepcopy(T)
                        Tc.remove(e)
                        child_node = Node(value=(Gc,Ec,Tc,e))

                        prob = probs[edges_ordered[e]].item()

                        children.append(child_node)
                        children_probs.append(prob)

                        if self.debug_mode == 3:
                            print("added edge", e)

                # normalize probabilities
                children_probs = self.normalize_probs(children_probs)

                # select with prob
                stackNode.extend(list(np.random.choice(children, len(children), p=children_probs, replace=False)))
                break

            if len(E.edges()) == 0 and f_found_max_children == 0:
                # E is empty
                if self.debug_mode >= 2:
                    print("E is empty")
                self.add_winners(G, I, known_winners, stats)

        return tuple(stats.strategy), stats.num_nodes, False


    def safe_div(self, x, y):
        if y==0:
            return 0
        return x/y

    def edges2string(self, edges, I):
        m = len(I)
        gstring = list(str(0).zfill(m**2))
        for e in edges:
            gstring[(e[0] - min(I))*m + e[1] - min(I)] = '1'

        return ''.join(gstring)

    def string2edges(self, gstring, I):
        m = len(I)
        edges = []
        for i in range(len(gstring)):
            if gstring[i] == '1':
                e1 = i % m + min(I)
                e0 = int((i - e1) / m) + min(I)
                edges.append((e0, e1))
        return edges


    def normalize_probs(self, probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

class Node:
    def __init__(self, value=None):
        self.value = value

    def __lt__(self, other):
        return 0

    def getvalue(self):
        return self.value