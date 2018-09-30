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




class MechanismRankedPairs():
    """
    The Ranked Pairs mechanism.
    """

    # debug_mode
    # = 0: no output
    # = 1: outputs only initial state
    # = 2: outputs on stop conditions
    # = 3: outputs all data
    def __init__(self):
        global debug_mode, BEGIN
        self.debug_mode = 0
        self.BEGIN = time.perf_counter()

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
    def state_features(self, G, E, I, K, T, a):
        u = a[0]
        v = a[1]

        f = []

        num_polynomial = 4

        # out/in degree
        f.extend(self.polynomialize(self.safe_div(G.out_degree(u), self.E_0.out_degree(u)), num_polynomial))
        f.extend(self.polynomialize(self.safe_div(G.in_degree(u), self.E_0.in_degree(u)), num_polynomial))
        f.extend(self.polynomialize(self.safe_div(G.out_degree(v), self.E_0.out_degree(v)), num_polynomial))
        f.extend(self.polynomialize(self.safe_div(G.in_degree(v), self.E_0.in_degree(v)), num_polynomial))

        # total degree
        f.extend(self.polynomialize(self.safe_div(G.out_degree(u) + G.in_degree(u), self.E_0.out_degree(u) + self.E_0.in_degree(u)), num_polynomial))
        f.extend(self.polynomialize(self.safe_div(G.out_degree(v) + G.in_degree(v), self.E_0.out_degree(v) + self.E_0.in_degree(v)), num_polynomial))

        # binary "has out/in degree" features
        f.append(2 * int(G.out_degree(u) > 0) - 1)
        f.append(2 * int(G.in_degree(u) > 0) - 1)
        f.append(2 * int(G.out_degree(v) > 0) - 1)
        f.append(2 * int(G.in_degree(v) > 0) - 1)

        # known winners features
        f.append(2 * int(u in K) - 1)
        f.append(2 * int(v in K) - 1)

        # if a in self.edge_to_cycle_occurrence:
        #     f.extend(self.polynomialize(self.safe_div(self.edge_to_cycle_occurrence[a], self.num_cycles), 1))
        # else:
        #     f.extend(self.polynomialize(0, 1))

        # voting rules scores
        f.extend(self.polynomialize(self.plurality_scores[u], num_polynomial))
        f.extend(self.polynomialize(self.plurality_scores[v], num_polynomial))
        f.extend(self.polynomialize(self.borda_scores[u], num_polynomial))
        f.extend(self.polynomialize(self.borda_scores[v], num_polynomial))
        f.extend(self.polynomialize(self.copeland_scores[u], num_polynomial))
        f.extend(self.polynomialize(self.copeland_scores[v], num_polynomial))
        f.extend(self.polynomialize(self.maximin_scores[u], num_polynomial))
        f.extend(self.polynomialize(self.maximin_scores[v], num_polynomial))

        f.extend(self.vectorized_wmg)
        f.extend(self.posmat)

        # visited feature (always will be 1 since caching)
        # f.extend([1])

        # edge weight
        f.extend(self.polynomialize(self.E_0[u][v]['weight'] / self.max_edge_weight, num_polynomial))

        # adjacency matrix if a is added
        G.add_edge(u,v)
        adjacency = nx.adjacency_matrix(G, nodelist = I).todense()
        adjacency = np.multiply(adjacency, self.adjacency_0)
        adjacency_normalized = np.divide(adjacency, 10) # NOTE: update if not using n10
        f.extend(adjacency_normalized.flatten().tolist()[0])
        G.remove_edge(u,v)

        # K representation
        K_list = []
        for i in I:
            if i in K:
                K_list.append(1)
            else:
                K_list.append(0)
        f.extend(K_list)

        # tier adjacency matrix
        legal_actions = T.copy()
        T_matrix = np.zeros((10, 10)) # needs to change for anything not 10x10
        for (c1,c2) in legal_actions:
            T_matrix[c1,c2] = 1
        T_vec = list(T_matrix.flatten())
        f.extend(T_vec)

        # edge connectivity
        f.extend(self.polynomialize(avg_edge_connectivity(G, I, u), num_polynomial))
        f.extend(self.polynomialize(avg_edge_connectivity(G, I, v), num_polynomial))

        # node connectivity
        f.extend(self.polynomialize(avg_node_connectivity(G, I, u), num_polynomial))
        f.extend(self.polynomialize(avg_node_connectivity(G, I, v), num_polynomial))

        # node2vec every time
        # G_with_weights = nx.DiGraph()
        # G_with_weights.add_nodes_from(self.I)
        # for (cand1, cand2) in self.G.edges():
        #     G_with_weights.add_edge(cand1, cand2, weight=self.E_0_really[cand1][cand2]['weight'])
        # node2vec_G = node2vec.Graph(G_with_weights, True, self.node2vec_args.p, self.node2vec_args.q)
        # node2vec_G.preprocess_transition_probs()
        # walks = node2vec_G.simulate_walks(self.node2vec_args.num_walks, self.node2vec_args.walk_length)
        # self.node2vec_model = node2vecmain.learn_embeddings(walks, self.node2vec_args)

        # node2vec features
        # node2vec_u = self.node2vec_model.wv[str(u)]
        # node2vec_v = self.node2vec_model.wv[str(v)]
        # node2vec_uv = np.append(node2vec_u, node2vec_v)

        # node2vec_f = np.append(node2vec_uv, np.array(f))

        if self.debug_mode >= 3:
            print("features", f)

        return Variable(torch.from_numpy(np.array(f)).float())
        # return Variable(torch.from_numpy(node2vec_f).float())



    def outer_loop_lp(self, profile, model, prediction=[]):
        """
        Returns 1. a list of all PUT-winners of profile under ranked pairs rule
        and 2. A Stats object of runtime statistics

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Initialize
        stats = self.Stats()

        wmg = profile.getWmg()
        known_winners = set()
        I = list(wmg.keys())

        # save profile as matrix for use in features
        profile_matrix = []
        for p in profile.preferences:
            profile_matrix.append(p.getOrderVector())
        self.profile_matrix = np.asmatrix(profile_matrix)

        G = nx.DiGraph()
        G.add_nodes_from(I)

        E = nx.DiGraph()
        E.add_nodes_from(I)
        self.max_edge_weight = 0
        for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
            if wmg[cand1][cand2] > 0:
                E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])
                self.max_edge_weight = max(self.max_edge_weight, wmg[cand1][cand2])

        self.E_0 = E.copy()
        self.adjacency_0 = nx.adjacency_matrix(E, nodelist=I).todense()

        # compute voting rules scores
        self.plurality_scores = plurality_score(self.profile_matrix)
        self.borda_scores = borda_score(self.profile_matrix)
        self.copeland_scores = copeland_score(wmg)
        self.maximin_scores = maximin_score(wmg)

        self.vectorized_wmg = vectorize_wmg(wmg)
        self.posmat = profile2posmat(self.profile_matrix)

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


        # if using cycle feature
        start_cycles = time.perf_counter()

        cycles = nx.simple_cycles(Gc)
        self.num_cycles = 0
        self.edge_to_cycle_occurrence = {}
        for c in cycles:
            self.num_cycles += 1
            for i in range(len(c) - 1):
                e0 = c[i]
                e1 = c[i + 1]
                e = (e0, e1)

                if e not in self.edge_to_cycle_occurrence:
                    self.edge_to_cycle_occurrence[e] = 0
                self.edge_to_cycle_occurrence[e] += 1

            e = (c[-1], c[0])
            if e not in self.edge_to_cycle_occurrence:
                self.edge_to_cycle_occurrence[e] = 0
            self.edge_to_cycle_occurrence[e] += 1

        stats.time_for_cycles += time.perf_counter() - start_cycles


        # Each node contains (G, E, tier)
        # root = Node(value=(self.edges2string(G.edges(), I), self.edges2string(E.edges(), I)))
        root = Node(value=(G, E, []))
        stackNode = []
        stackNode.append(root)

        hashtable = set()

        END = self.BEGIN + self.TIMEOUT

        while stackNode:
            # Pop new node to explore
            node = stackNode.pop()
            (G, E, T) = node.value

            if time.perf_counter() > END:
                print("TIMEOUT")
                return sorted(known_winners), stats

            # Check hash
            hash_state = self.edges2string(G.edges(), I) + self.edges2string(E.edges(), I) + self.edges2string(T, I)
            if hash_state in hashtable:
                stats.num_hashes += 1
                if self.debug_mode == 3:
                    print("hashed in outer hashtable")
                continue
            hashtable.add(hash_state)

            stats.num_nodes += 1

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

                # Used to break ties
                index = 0

                # Add each edge to stack by priority
                children = dict()
                T = sorted(T)
                for e in T:
                    if not G_tc.has_edge(e[1], e[0]):
                        f_found_max_children = 1

                        Gc = G.copy()
                        Gc.add_edges_from([e])
                        Ec = E.copy()
                        Tc = copy.deepcopy(T)
                        Tc.remove(e)
                        child_node = Node(value=(Gc,Ec,Tc))

                        # LPwinners
                        # G_in_degree = Gc.in_degree(I)
                        # potential_winners = set([x[0] for x in G_in_degree if x[1] == 0])
                        # priority = len(potential_winners - known_winners)

                        EUT = E.copy()
                        EUT.add_edges_from(T)

                        priority = model(self.state_features(G, EUT, I, known_winners, Tc, e))

                        children[child_node] = (priority, index)
                        index = index + 1

                        if self.debug_mode == 3:
                            print("added edge", e)

                children_items = sorted(children.items(), key=lambda x: (x[1][0], x[1][1]))
                sorted_children = [key for key, value in children_items]
                stackNode += sorted_children
                break

            if len(E.edges()) == 0 and f_found_max_children == 0:
                # E is empty
                if self.debug_mode >= 2:
                    print("E is empty")
                self.add_winners(G, I, known_winners, stats)

        return sorted(known_winners), stats


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


class Node:
    def __init__(self, value=None):
        self.value = value

    def __lt__(self, other):
        return 0

    def getvalue(self):
        return self.value

def read_Y_result(inputfile):
    Y = dict()
    temp = inputfile.readline()
    filenames = []
    while temp:
        infomation = temp.strip().split(" =")
        filenames.append(infomation[0])
        # print("%s！"% infomation[0])
        x = infomation[1].split()
        # print(x)
        # x = x.split(', ')
        # print(x)
        Y[infomation[0]] = [ int( x ) for x in x if x ]
        # print(Y[infomation[0]])
        temp = inputfile.readline()

    return Y, filenames


def read_Y_prediction(inputfile):
    Y = dict()
    temp = inputfile.readline()
    filenames = []
    while temp:
        infomation = temp.strip().split(":")
        filenames.append(infomation[0])
        # print(infomation[1])
        x = infomation[1].split()
        # print(x)
        # x = x.split(', ')
        # print(x)
        Y[infomation[0]] = [ float( x ) for x in x if x ]
        # print(Y[infomation[0]])
        temp = inputfile.readline()

    return Y, filenames


if __name__ == '__main__':
    def handler(signum, frame):
        raise AssertionError

    os.chdir(rpconfig.path)
    y1_filenames = rpconfig.filename
    inf1 = open(y1_filenames, 'r')
    prediction, filenames = read_Y_prediction(inf1)
    inf1.close()

    num_profiles = 0
    total_time = 0
    total_node = 0
    total_100time = 0
    total_100node = 0
    total_hits = 0
    total_hash = 0
    # filenames = ['M10N10-16903.csv']
    filenames = filenames[0:1000]

    # filenames = sorted(glob.glob('M5N5-*.csv'))

    output_filename = "results_" + os.path.basename(__file__)
    output_filename = output_filename.replace('.py','')
    output_filename += '.txt'

    output_file = open(output_filename, "w+")

    print(os.path.basename(__file__))
    header = "inputfile\tPUT-winners\tnum nodes\tdiscovery states\tmax discovery state\tdiscovery times\tmax discovery times\tstop condition hits\tsum stop cond hits\tnum hashes\tnum initial bridges\tnum redundant edges\truntime"
    print(header)
    output_file.write(header+'\n')


    # load RL model
    D_in = 482
    H1 = 1000
    H2 = 1000
    D_out = 1
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H1, H2),
        torch.nn.Sigmoid(),
        torch.nn.Linear(H2, D_out)
    )

    checkpoint_filename = "C:\\Users\\shepht2\\Documents\\School\\Masters\\STV Ranked Pairs\\RL\\results\\9-22\\results_RP_RL_main240295240_model.pth.tar"
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint)
    print("loaded model from", checkpoint_filename)
    output_file.write("loaded model from " + str(checkpoint_filename) + '\n')

    for inputfile in filenames:
        # try:
        #     signal.signal(signal.SIGALRM, handler)
        #     signal.alarm(3)

        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()

        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "soi" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        # Skip the really really hard cases
        # if inputfile == "M20N20-100046.csv" or inputfile == "M20N20-100088.csv":
        #     print("%s\tskipped" % (inputfile))
        #     continue

        start = time.perf_counter()
        rp_results = MechanismRankedPairs().outer_loop_lp(profile, model, prediction=prediction[inputfile])
        end = time.perf_counter()

        PUT_winners = rp_results[0]
        stats = rp_results[1]

        if len(stats.discovery_states.values()) > 0:
            max_discovery_state = max(stats.discovery_states.values())
        else:
            max_discovery_state = -1

        if len(stats.discovery_times.values()) > 0:
            max_discovery_time = max(stats.discovery_times.values())
        else:
            max_discovery_time = - 1

        num_stop_condition_hits = sum(list(stats.stop_condition_hits.values()))

        result_text = "%s\t%r\t%d\t%r\t%d\t%r\t%f\t%r\t%d\t%d\t%d\t%d\t%f\t%f" % (inputfile, PUT_winners, stats.num_nodes, stats.discovery_states,
                                                  max_discovery_state, stats.discovery_times, max_discovery_time, stats.stop_condition_hits, num_stop_condition_hits, stats.num_hashes, stats.num_initial_bridges, stats.num_redundant_edges, stats.time_for_cycles, (end - start))

        print(result_text)
        output_file.write(result_text + '\n')
        output_file.flush()

        num_profiles += 1
        total_time += end - start
        total_node += stats.num_nodes
        total_100time += max_discovery_time
        total_100node += max_discovery_state
        total_hits += num_stop_condition_hits
        total_hash += stats.num_hashes

        # signal.alarm(0)
        # except AssertionError:
        #     print("timeout")
    ave_time = total_time / num_profiles
    ave_node = total_node / num_profiles
    ave_100time = total_100time / num_profiles
    ave_100node = total_100node / num_profiles
    ave_hits = total_hits / num_profiles
    ave_hash = total_hash / num_profiles
    print("#profiles %f\n#avg_node %f\n#avg_100node %f\navg_time %f\navg_100time %f\navg_hits %f\navg_hash %f\n" % (num_profiles, ave_node, ave_100node, ave_time, ave_100time, ave_hits, ave_hash))

    output_file.close()