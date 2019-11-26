#!/usr/bin/env python

import math
import networkx as nx
import pickle
import sys
import os
from numpy import random

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib as plt
    plt.use('TkAgg')
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import combinations
from graph_stats import calc_stats, plot

PAGERANK_MEMO = {}


def usage(code):
    print("Usage: compression.py [filename]")
    exit(code)


def make_all_dirs(dir_list):
    for d in dir_list:
        if not os.path.isdir('./' + d):
            os.mkdir('./' + d)


def read_graph(filename):
    g = {}
    changes = defaultdict(lambda: nx.MultiDiGraph())
    current_graph = nx.DiGraph()
    previous_timestamp = -1
    with open(filename, 'r') as f:
        for line in f:
            # DBLP
            source, target, weight, timestamp = line.strip().split()
            # DARPA
            #timestamp, source, target, weight = line.split()
            changes[timestamp].add_edge(source, target, weight=int(weight))

    # we don't want to store only added edges at each timestep, but the current graph snapshot
    for timestamp, graph in sorted(changes.items()):
        current_graph.add_edges_from(graph.edges(data=True))
        g[timestamp] = current_graph.copy()

    return g


def lambda_connection(graph, source, target, lambda_=5):
    # RWR
    #if graph in PAGERANK_MEMO:
    #    return PAGERANK_MEMO[graph][target]

    weights = nx.pagerank(graph, personalization={source: 1}, tol=1e-03)
    #PAGERANK_MEMO[graph] = weights
    return weights[target]


def lambda_distance(graph1, graph2, source, target):
    # both graphs have to have the same vertex set
    # note: order in NodeView may be different
    assert set(graph1.nodes) == set(graph2.nodes)

    source_nodes, target_nodes = [], []
    for s in source.split('+'):
        for node in two_hop_pairs(graph2, s):
            source_nodes.append(node)
    for t in target.split('+'):
        for node in two_hop_pairs(graph2, t):
            target_nodes.append(node)


    total = 0
    for s, t in zip(random.permutation(source_nodes), random.permutation(target_nodes)): # random pertumation???
    #for s, t in zip(source_nodes, target_nodes): # random pertumation???
        total += (lambda_connection(graph1, s, t) - lambda_connection(graph2, s, t))**2
    return math.sqrt(total)


def decompress(compressed_graph):
    # there's an edge between two nodes exactly when theres a superedge between corr supernodes
    # weight of an edge is the weight of the corresponding supernode
    graph = compressed_graph.copy()
    # recreate original node set in 'graph'
    node_to_supernode = {}
    for supernode in compressed_graph.nodes:
        #nodes = compressed_graph.nodes.data()[supernode]['contains']
        nodes = supernode.split('+')
        graph.remove_node(supernode)
        for node in nodes:
            graph.add_node(node, contains=set([node]))
            node_to_supernode[node] = supernode

    # add edges in G corresponding to superedges in S
    for source, target in compressed_graph.edges:
        contains = nx.get_node_attributes(compressed_graph, 'contains')
        superweight = compressed_graph[source][target]['weight']
        for s in source.split('+'):
            for t in target.split('+'):
                graph.add_edge(s, t, weight=superweight)

    return graph


def is_edge(g, tup):
    return tup in g.edges

def is_supernode_neighbor(supersrc, supertgt, s, t):
    return s in (supersrc, supertgt) or t in (supersrc, supertgt)


def graph_merge(graph, source, target):
    contains = nx.get_node_attributes(graph, 'contains')

    def W(s, t):
        if s != t:
            scale = len(contains[s]) * len(contains[t])
        else:
            scale = len(contains[s]) * (len(contains[s]) - 1) / 2
        return lambda_connection(graph, s, t) * scale

    new_graph = graph.copy()
    new_graph.remove_node(source)
    new_graph.remove_node(target)

    # create new supernode, add to graph
    supernode = '{}+{}'.format(source, target)
    supernode_contains = supernode.split('+')
    contains[source] = source.split('+')
    contains[target] = target.split('+')
    new_graph.add_node(supernode, contains=supernode_contains)

    # update weighted edges accordingly
    for (u, v) in graph.edges:
        if '{}+{}'.format(u, v) == supernode or '{}+{}'.format(v, u) == supernode:
            continue
        if u == v:
            continue
        if source == v or target == v and (u, v) != (source, target):
            node = v
            num =  len(contains[source]) * lambda_connection(graph, source, node)
            num += len(contains[target]) * lambda_connection(graph, target, node)
            denom = len(contains[source]) + len(contains[target])
            new_graph.add_edge(u, supernode, weight=num/denom)
            print(supernode, u, v)

        elif source == u or target == u and '{}+{}'.format(u, v) != supernode:
            node = u
            num =  len(contains[source]) * lambda_connection(graph, source, node)
            num += len(contains[target]) * lambda_connection(graph, target, node)
            denom = len(contains[source]) + len(contains[target])
            new_graph.add_edge(supernode, v, weight=num/denom)

    # update w'({z, z})
    num = W(source, source) + W(target, target) + W(source, target)
    denom = len(supernode_contains) * (len(supernode_contains) - 1) / 2
    new_graph.add_edge(supernode, supernode, weight=num/denom)

    return new_graph


def init_compressed_graph(graph):
    new_graph = graph.copy()
    attributes = {node: {'contains': set([node])} for node in graph.nodes}
    nx.set_node_attributes(new_graph, attributes)
    return new_graph


def calc_cr(graph, compressed_graph):
    return compressed_graph.number_of_nodes() / graph.number_of_nodes()


def two_hop_pairs(g, source):
    assert source in g.nodes
    seen = set()
    for intermediate in g.neighbors(source):
        if source == intermediate or intermediate in seen:
            continue
        seen.add(intermediate)
        yield intermediate
        for target in g.neighbors(intermediate):
            if source == target or target in seen:
                continue
            #seen.append(target)
            yield target
    #return seen


def brute_force_greedy(graph, cr=0.1, min_distance=0.5):
    compressed_graph = init_compressed_graph(graph)
    for _ in range(10):
        has_merged = False
        for source in random.permutation(list(compressed_graph.nodes)):
        #for source in compressed_graph.nodes:
            if source not in compressed_graph.nodes:
                continue
            for target in two_hop_pairs(compressed_graph, source):
                # check that the node hasn't already been merged on prev. iteration
                if source not in compressed_graph.nodes or target not in compressed_graph.nodes:
                    continue
                if source == target:
                    continue
                temp_graph = graph_merge(compressed_graph, source, target)
                distance = lambda_distance(graph, decompress(temp_graph), source, target)
                print(source, target, distance)
                if distance < min_distance:
                    compressed_graph = temp_graph
                    has_merged = True
                    if calc_cr(graph, compressed_graph) < cr:
                        return compressed_graph
        if not has_merged:
            print('Nothing left to merge')
            break

    return compressed_graph


if __name__ == '__main__':
    # argument parsing
    if len(sys.argv) != 2:
        usage(1)

    filename = sys.argv[1].strip()
    make_all_dirs(['pkl_files', 'plots'])

    compressed_g = {}
    g =  read_graph(filename)
    for timestamp, graph in g.items():
        print('timestamp', timestamp)
        compressed = brute_force_greedy(graph)
        print('  final cr  : ', calc_cr(graph, compressed))
        '''
        print('  nodes:')
        for node in compressed:
            print(node)
        print()
        '''
        compressed_g[timestamp] = compressed

        prefix = filename.split('/')[-1].split('.')[0]
        with open('./pkl_files/{}_compressed.pkl'.format(prefix), 'wb') as f:
            pickle.dump(compressed_g, f)
    prefix = filename.split('/')[-1].split('.')[0]
    stats = calc_stats(g)
    plot(g, stats, prefix)
