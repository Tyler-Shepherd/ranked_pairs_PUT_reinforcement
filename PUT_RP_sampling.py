import io
import os
import prefpy_io
import math
import time
from numpy import *
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
import random
import glob
from pprint import pprint
import torch

from RP_RL_agent import RP_RL_agent
import params as params
import RP_utils
from RL_base import RL_base

# samples using agent then runs full PUT_RP

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
            self.num_outer_nodes = 0
            self.stop_condition_hits = {1: 0, 2: 0, 3: 0, 4: 0}
            self.num_hashes = 0
            self.num_initial_bridges = 0
            self.num_redundant_edges = 0
            self.num_sampled = 0
            self.sampled = []

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
                if self.debug_mode >= 2:
                    print("Found new winner:", c)

    def stop_conditions(self, G, E, I, known_winners, stats):
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
            if self.debug_mode >= 2:
                print("Stop Condition 2: pruned")
            return 2

        # Stop Condition 3: Exactly one node has indegree 0
        if len(possible_winners) == 1:
            stats.stop_condition_hits[3] += 1
            if self.debug_mode >= 2:
                print("Stop Condition 3: one cand in degree 0")
            self.add_winners(G, I, known_winners, stats, possible_winners)
            return 3

        # Stop Condition 1: G U E is acyclic
        temp_G = nx.compose(G, E)
        if nx.is_directed_acyclic_graph(temp_G) is True:
            stats.stop_condition_hits[1] += 1
            if self.debug_mode >= 2:
                print("Stop Condition 1: acyclic")
            self.add_winners(G, I, known_winners, stats)
            return 1

        return -1

    def outer_loop_lp(self, profile, agent):
        """
        Returns 1. a list of all PUT-winners of profile under ranked pairs rule
        and 2. A Stats object of runtime statistics

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Sample
        # number of samples is defined by params.num_testing_iterations
        # since sampling is just testing v1
        known_winners, iters_discovered, _, times_discovered = agent.test_model(profile)

        # Initialize
        stats = self.Stats()
        stats.discovery_times = times_discovered
        for c in known_winners:
            stats.discovery_states[c] = 0

        stats.num_sampled = len(known_winners)
        stats.sampled = sorted(known_winners.copy())

        G = agent.G_0
        E = agent.E_0
        I = agent.I

        # Each node contains (G, E)
        root = Node(value=(G, E))
        stackNode = []
        stackNode.append(root)

        hashtable = set()

        while stackNode:
            # Pop new node to explore
            node = stackNode.pop()
            (G, E) = node.value

            # Check hash
            hash_state = hash(str(G.edges()) + str(E.edges()))
            if hash_state in hashtable:
                stats.num_hashes += 1
                if self.debug_mode == 3:
                    print("hashed in outer hashtable")
                continue
            hashtable.add(hash_state)

            stats.num_outer_nodes += 1
            stats.num_nodes += 1

            if self.debug_mode == 3:
                print("Popped new node: ")
                print("G:", G.edges())
                print("E:", E.edges())

            # Flag for whether expanding the current tier required finding max children
            f_found_max_children = 0

            # Continue performing RP on this state as long as tie-breaking order doesn't matter
            while len(E.edges()) != 0:
                if self.stop_conditions(G, E, I, known_winners, stats) != -1:
                    # Stop condition hit
                    break

                (max_weight, max_edge) = max([(d['weight'], (u, v)) for (u, v, d) in E.edges(data=True)])
                ties = [d['weight'] for (u, v, d) in E.edges(data=True)].count(max_weight)

                if ties == 1:
                    # Tier only has one edge
                    if self.debug_mode == 3:
                        print("Only 1 edge in tier")

                    E.remove_edges_from([max_edge])
                    if nx.has_path(G, max_edge[1], max_edge[0]) is False:
                        G.add_edges_from([max_edge])

                else:
                    # This tier has multiple edges with same max weight.
                    tier = [(u, v) for (u, v, d) in E.edges(data=True) if d['weight'] == max_weight]
                    if self.debug_mode == 3:
                        print("Tier =", tier)

                    E.remove_edges_from(tier)

                    # Compute "bridge edges" which are not in any cycle
                    Gc = G.copy()
                    Gc.add_edges_from(tier)
                    scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if
                           len(g.edges()) != 0]
                    bridges = set(Gc.edges()) - set(itertools.chain(*scc))
                    G.add_edges_from(bridges)
                    tier = list(set(tier) - bridges)

                    G_tc = nx.transitive_closure(G)

                    # Remove "inconsistent edges" that cannot be added to G without causing cycle
                    reverse_G = nx.DiGraph.reverse(G_tc)
                    tier = list(set(tier) - set(reverse_G.edges()))

                    # Remove "redundant edges": if there is already path from e[0] to e[1], can immediately add e
                    redundant_edges = set()
                    for e in tier:
                        if G_tc.has_edge(e[0], e[1]):
                            redundant_edges.add(e)
                            G.add_edges_from([e])
                    stats.num_redundant_edges += len(redundant_edges)
                    tier = list(set(tier) - redundant_edges)

                    if len(tier) == 0:
                        # No need to find max children, as tier is now empty
                        continue

                    max_children = self.find_max_children_scc_decomposition(G, tier, scc, bridges, I, known_winners, stats)

                    # Determine priority ordering of maximal children
                    children = dict()
                    index = 0
                    for child in max_children:
                        # child_node = Node(value=(self.edges2string(child.edges(), I), self.edges2string(E.edges(), I)))
                        child_node = Node(value=(child, E.copy()))
                        c_in_deg = child.in_degree(I)
                        available = set([x[0] for x in c_in_deg if x[1] == 0])
                        priority = len(available - known_winners)
                        # children[child_node] = (priority, index)
                        children[child_node] = index
                        child.add_nodes_from(I)
                        index += 1
                        continue

                    children_items = sorted(children.items(), key=lambda x: x[1])
                    sorted_children = [key for key, value in children_items]
                    stackNode += sorted_children
                    f_found_max_children = 1
                    break

            # f_found_max_children is needed since, if we just added more nodes to stack, then current (G, E) is not actual valid state
            if len(E.edges()) == 0 and f_found_max_children == 0:
                # E is empty
                if self.debug_mode >= 2:
                    print("E is empty")
                self.add_winners(G, I, known_winners, stats)

        return sorted(known_winners), stats, iters_discovered

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

    def find_max_children_scc_decomposition(self, G, tier, scc, bridges, I, known_winners, stats):
        '''
        Finds the maximal children of G when tier is added using SCC decomposition
        :param G: Networkx DiGraph of edges "locked in" so far
        :param tier: List of edges in the current tier to be added with equal weight
        :param scc: List of the strongly connected components of G U tier, each being a list of edges
        :param bridges: List of edges that are bridges between sccs of G U tier
        :param I: List of all nodes
        :param known_winners: Known PUT-winners computed by RP so far
        :param stats: Stats object containing runtime statistics
        :return: Array of Networkx DiGraphs that are the maximal children of G U T
        '''

        if len(scc) == 1:
            children = self.explore_max_children_lp(G, tier, I, known_winners, stats)
            return children

        mc_list = []

        for x in scc:
            G_temp = nx.DiGraph(list(set(G.edges()).intersection(set(x))))
            T_temp = list(set(tier).intersection(set(x)))
            temp = self.explore_max_children_lp(G_temp, T_temp, I, known_winners, stats, f_scc = 1)
            mc_list.append(temp)

        Cartesian = itertools.product(*mc_list)
        return [nx.DiGraph(list(set(itertools.chain(*[list(y.edges()) for y in x])).union(bridges))) for x in Cartesian]


    def explore_max_children_lp(self, G, tier, I, known_winners, stats, f_scc = 0):
        """
        Computes the maximal children of G when tier is added
        :param G: DiGraph, A directed graph
        :param tier: list of tuples which correspond to multiple edges with same max weight.
                    e.g. edges = [x for x in wmg2.keys() if wmg2[x] == max_weight]
        :param I: all nodes in G
        :param known_winners: PUT-winners found so far by RP
        :param stats: Stats object
        :param f_scc: set to 1 if the G and tier being considered are an SCC of the full graph due to SCC decomposition
        :return: set of graphs which correspond to maximum children of given parent: G
        """

        # self.output_graph(G)
        # self.output_graph(nx.DiGraph(tier))

        max_children = []
        cstack = []

        # print("start mc:", time.perf_counter() - self.BEGIN)

        hashtable = set()

        if self.debug_mode >= 1:
            print("Exploring max children")
            print("G:", G.edges())
            print("Tier:", tier)
            print("Known winners:", known_winners)
            print("---------------------------")

        in_deg = G.in_degree()
        nodes_with_no_incoming = set()
        for x in in_deg:
            if x[1] == 0:
                nodes_with_no_incoming.add(x[0])
        for x in I:
            if x not in G.nodes():
                nodes_with_no_incoming.add(x)

        root = Node(value=(self.edges2string(G.edges(), I), self.edges2string(tier, I), nodes_with_no_incoming))
        cstack.append(root)

        END = self.BEGIN + self.TIMEOUT

        while cstack:
            node = cstack.pop()
            (G_str, T_str, no_incoming) = node.value

            if time.perf_counter() > END:
                print("TIMEOUT")
                return max_children

            # Check hash. Doesn't ever happen if the below hash is included
            hash_G = hash(G_str)
            if hash_G in hashtable:
                stats.num_hashes += 1
                print('hash')
                if self.debug_mode >= 2:
                    print("hashed in hashtable")
                continue
            hashtable.add(hash_G)

            stats.num_nodes += 1

            G = nx.DiGraph(self.string2edges(G_str, I))
            T = self.string2edges(T_str, I)
            G.add_nodes_from(I)

            if self.debug_mode == 3:
                print("popped")
                print("G: ", G.edges())
                print("T: ", T)

            # goal state 2: if current G's possible winners is subset of known winners,
            # then directly ignore it.
            if no_incoming <= known_winners and not f_scc:
                stats.stop_condition_hits[2] += 1
                if self.debug_mode >= 3:
                    print("MC goal state 2: pruned")
                continue

            # goal state 1: if there are no edges to be added, then add the G_
            if len(T) == 0:
                max_children.append(G.copy())
                if self.debug_mode >= 2:
                    print("MC goal state 1: no more edges in tier")
                    print("max child: ", G.edges())
                continue

            # goal state 3: if current G has exactly one cand with in degree 0, it is a PUT-winner
            if len(no_incoming) == 1 and not f_scc:
                stats.stop_condition_hits[3] += 1
                if self.debug_mode >= 2:
                    print("MC goal state 3: only one cand in degree 0")
                    print("max child:", G.edges())
                self.add_winners(G, I, known_winners, stats, no_incoming)
                continue

            # goal state 4: if union of current G and edges is acyclic,
            # then directly add it to the max_children_set
            Gc = G.copy()
            Gc.add_edges_from(T)
            if nx.is_directed_acyclic_graph(Gc):
                stats.stop_condition_hits[4] += 1

                hash_temp_G = hash(self.edges2string(Gc.edges(), I))
                if hash_temp_G not in hashtable:
                    hashtable.add(hash_temp_G)
                    max_children.append(Gc)

                    if self.debug_mode >= 2:
                        print("MC goal state 4: G U T is acyclic")
                        print("max child:", Gc.edges())
                else:
                    stats.num_hashes += 1
                continue

            # Perform reductions every step:

            # Compute "bridge edges" which are not in any cycle
            Gc = G.copy()
            Gc.add_edges_from(T)
            scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if
                   len(g.edges()) != 0]
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

            # Flag for whether adding any edge from T causes G to remain acyclic
            f_isAcyclic = 0

            children = dict()

            # Used to break ties
            index = 0
            T = sorted(T)
            for e in T:
                G.add_edges_from([e])
                Gc_str = self.edges2string(G.edges(), I)
                if hash(Gc_str) in hashtable:
                    f_isAcyclic = 1

                    stats.num_hashes += 1
                    G.remove_edges_from([e])
                    continue

                if not nx.has_path(G, source=e[1], target=e[0]):
                    f_isAcyclic = 1

                    Tc = copy.deepcopy(T)
                    Tc.remove(e)

                    # Remove the head of the edge if it had no incoming edges previously
                    no_incoming_c = no_incoming.copy()
                    no_incoming_c.discard(e[1])

                    child = Node(value=(Gc_str, self.edges2string(Tc, I), no_incoming_c))

                    priority = len(no_incoming_c - known_winners)

                    children[child] = (priority, index)
                    index = index + 1

                    if self.debug_mode == 3:
                        print("add new child with edge ", e, " and priority ", priority)

                G.remove_edges_from([e])

            children_items = sorted(children.items(), key=lambda x: (x[1][0], x[1][1]))
            sorted_children = [key for key, value in children_items]
            cstack += sorted_children

            # goal state 5: adding all edges in T individually cause G to be cyclic
            if f_isAcyclic == 0:
                max_children.append(G.copy())

                if self.debug_mode >= 2:
                    print("MC goal state 5 - found max child")
                    print("max child: ", G.edges())
                continue

        if self.debug_mode >= 1:
            print("finished exploring max children")
            print("num max children:", len(max_children))
            print("PUT-winners:", known_winners)

        return max_children


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

    filenames = filenames[0:1000]

    random.seed(5)

    # load model
    model = torch.nn.Sequential(
        torch.nn.Linear(params.D_in, params.H1),
        torch.nn.Sigmoid(),
        torch.nn.Linear(params.H1, params.D_out)
    )

    if params.f_start_from_default:
        RP_utils.load_model(model, params.default_model_path)
    else:
        print('***************NOT STARTING FROM DEFAULT*********************')

    agent = RP_RL_agent(model)

    print("inputfile\tPUT-winners\tnum nodes\tdiscovery states\tmax discovery state\tdiscovery times\tmax discovery times\tstop condition hits\tsum stop cond hits\tnum hashes\tnum initial bridges\tnum redundant edges\tnum sampled\tsampled\tsamples discovered\tmax sample discovery\truntime")

    for inputfile in filenames:
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

        start = time.perf_counter()
        rp_results = MechanismRankedPairs().outer_loop_lp(profile, agent)
        end = time.perf_counter()

        PUT_winners = rp_results[0]
        stats = rp_results[1]
        samples_discovered = rp_results[2]
        max_discovery_state = max(stats.discovery_states.values())
        max_discovery_time = max(stats.discovery_times.values())
        num_stop_condition_hits = sum(list(stats.stop_condition_hits.values()))
        max_sample = max(samples_discovered.values())

        print("%d\t%s\t%r\t%d\t%r\t%d\t%r\t%f\t%r\t%d\t%d\t%d\t%d\t%d\t%r\t%r\t%d\t%f" % (num_profiles, inputfile, PUT_winners, stats.num_nodes, stats.discovery_states,
                                                  max_discovery_state, stats.discovery_times, max_discovery_time, stats.stop_condition_hits, num_stop_condition_hits, stats.num_hashes, stats.num_initial_bridges, stats.num_redundant_edges, stats.num_sampled, stats.sampled, samples_discovered, max_sample, (end - start)))

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