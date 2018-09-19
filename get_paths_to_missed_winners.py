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
import networkx.algorithms.connectivity
import glob
from pprint import pprint


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

        self.missed_winners = set()

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

    def add_winners(self, G, P, I, known_winners, stats, possible_winners = None):
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
            if c in self.missed_winners:
                # add each data point from P
                # never add c to K
                for item in P:
                    self.data[c].append(item)
            elif c not in known_winners:
                known_winners.add(c)
                stats.discovery_states[c] = stats.num_nodes
                stats.discovery_times[c] = time.perf_counter() - self.BEGIN

                if self.debug_mode >= 1:
                    print("Found new winner:", c)
                    print(G.edges())

    def stop_conditions(self, G, E, T, P, I, known_winners, stats):
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
            self.add_winners(G, P, I, known_winners, stats, possible_winners)
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

    def outer_loop_lp(self, profile, missed_winners):
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

        G = nx.DiGraph()
        G.add_nodes_from(I)

        E = nx.DiGraph()
        E.add_nodes_from(I)
        for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
            if wmg[cand1][cand2] > 0:
                E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])

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


        # Each node contains (G, E, T, P)
        # P is path, where each item is of form (G, E, K, a)
        # root = Node(value=(self.edges2string(G.edges(), I), self.edges2string(E.edges(), I)))
        root = Node(value=(G, E, [], []))
        stackNode = []
        stackNode.append(root)

        hashtable = set()

        END = self.BEGIN + self.TIMEOUT

        self.missed_winners = set(missed_winners)

        self.data = {}
        for w in missed_winners:
            self.data[w] = []

        while stackNode:
            # Pop new node to explore
            node = stackNode.pop()
            (G, E, T, P) = node.value

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
                if self.stop_conditions(G, E, T, P, I, known_winners, stats) != -1:
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
                            P.append((self.edges2string(G.edges, I), self.edges2string(E.edges, I), known_winners.copy(), max_edge))
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
                        Pc = copy.deepcopy(P)

                        EUT = E.copy()
                        EUT.add_edges_from(Tc)
                        Pc.append((self.edges2string(Gc.edges, I), self.edges2string(EUT.edges, I), known_winners.copy(), e))
                        child_node = Node(value=(Gc,Ec,Tc,Pc))

                        # LPwinners
                        G_in_degree = Gc.in_degree(I)
                        potential_winners = set([x[0] for x in G_in_degree if x[1] == 0])
                        priority = len(potential_winners - known_winners)

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
                self.add_winners(G, P, I, known_winners, stats)

        return sorted(known_winners), stats, self.data


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


if __name__ == '__main__':
    missed_winners_profiles_file = open('profiles_missed_winners.txt', 'r')

    # has arrays of form [profile_name, missed winner 1, missed winner 2, ...]
    missed_winners = []

    for line in missed_winners_profiles_file:
        line = line.strip('\n')
        missed_winners.append([line])
    missed_winners_profiles_file.close()

    missed_winners_file = open('missed_winners.txt', 'r')

    i = 0
    for line in missed_winners_file:
        for c in line:
            if c != '[' and c != ']' and c != ' ' and c != ',' and c != '\n':
                missed_winners[i].append(int(c))

        i += 1
    missed_winners_file.close()

    os.chdir(rpconfig.path)

    output_data_file = open('missed_winners_data.dat', 'w')

    # need to update for anything other than m10
    I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for thing in missed_winners:
        inputfile = thing[0]
        profile_missed_winners = thing[1:]

        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()

        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        winners, stats, data = MechanismRankedPairs().outer_loop_lp(profile, profile_missed_winners)

        print(inputfile, winners)
        pprint(data)

        # TODO: need to loop over keys of data (which are the missed winners) and somehow store what missed winner each data point goes to
        # TODO: store profile

        # output to file
        # outputs in form G E K a[0] a[1]
        output_data_file.write(inputfile + '\n')
        for w in data.keys():
            data_points = data[w]
            output_data_file.write("winner" + '\t' + str(w) + '\n')
            for d in data_points:
                G_str = d[0]
                E_str = d[1]
                K_str = ""
                for i in I:
                    if i in d[2]:
                        K_str += '1'
                    else:
                        K_str += '0'
                u_str = str(d[3][0])
                v_str = str(d[3][1])
                output_data_file.write(G_str + '\t' + E_str + '\t' + K_str + '\t' + u_str + '\t' + v_str + '\n')

        output_data_file.flush()

    output_data_file.close()