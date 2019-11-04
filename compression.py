#!/usr/bin/env python

import math
import networkx as nx
import pickle
import sys

from collections import defaultdict
from itertools import combinations

# TODO: unit tests

def read_graph(filename):
    g = defaultdict(lambda: nx.DiGraph())
    with open(filename, 'r') as f:
        for line in f:
            source, target, weight, timestamp = line.split()
            g[timestamp].add_edge(source, target, weight=int(weight))

    return g


def lambda_connection(graph, source, target, lambda_=5):
    # using shortest path weight as quality function
    try:
        weight, path = nx.single_source_dijkstra(graph, source, target, cutoff=lambda_)
    except:
        return 0
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    return 1/weight if weight != 0 else 0


def lambda_distance(graph1, graph2):
    # both graphs have to have the same vertex set
    # note: order in NodeView may be different
    assert set(graph1.nodes) == set(graph2.nodes)

    total = 0
    for s, t in zip(graph1.nodes, graph2.nodes):
        total += (lambda_connection(graph1, s, t) - lambda_connection(graph2, s, t))**2

    return math.sqrt(total)


def decompress(compressed_graph):
    # there's an edge between two nodes exactly when theres a superedge between corr supernodes
    # weight of an edge is the weight of the corresponding supernode
    graph = compressed_graph.copy()
    # recreate original node set in 'graph'
    node_to_supernode = {}
    for supernode in compressed_graph.nodes:
        nodes = compressed_graph.nodes.data()[supernode]['contains']
        graph.remove_node(supernode)
        for node in nodes:
            graph.add_node(node, contains=set([node]))
            node_to_supernode[node] = supernode

    # add edges in G corresponding to superedges in S
    for source, target in compressed_graph.edges:
        contains = nx.get_node_attributes(compressed_graph, 'contains')
        superweight = compressed_graph[source][target]['weight']
        for s in contains[source]:
            for t in contains[target]:
                graph.add_edge(s, t, weight=superweight)

    return graph


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
    supernode_contains = contains[source].union(contains[target])
    new_graph.add_node(supernode, contains=supernode_contains)

    # update weighted edges accordingly
    for node in graph.nodes:
        if node in (source, target):
            continue
        num =  len(contains[source]) * lambda_connection(graph, source, node)
        num += len(contains[target]) * lambda_connection(graph, target, node)
        denom = len(contains[source]) + len(contains[target])
        new_graph.add_edge(supernode, node, weight=num/denom)

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


def all_two_hop_pairs(g):
    seen = set()
    for source in g.nodes:
        for intermediate in g.neighbors(source):
            for target in g.neighbors(intermediate):
                if source == target or (source, target) in seen:
                    continue
                seen.add((source, target))
                yield source, target



def brute_force_greedy(graph, cr=0.90):
    compressed_graph = init_compressed_graph(graph)
    while calc_cr(graph, compressed_graph) > cr:
        print(calc_cr(graph, compressed_graph))
        min_source, min_target = -1, -1
        min_distance = float('inf')
        for i, (source, target) in enumerate(all_two_hop_pairs(compressed_graph)):
            distance = lambda_distance(graph, decompress(compressed_graph))
            if distance < min_distance:
                min_source = source
                min_target = target
        if min_source == -1 and min_target == -1:
            print('couldnt compress any more')
            return compressed_graph
        compressed_graph = graph_merge(compressed_graph, min_source, min_target)

    return compressed_graph


if __name__ == '__main__':
    filename = sys.argv[1].strip()
    compressed_g = []
    g =  read_graph(filename)
    for timestamp, graph in g.items():
        print('timestamp', timestamp)
        compressed = brute_force_greedy(graph)
        for node in compressed:
            print(node)

        compressed_g.append(compressed)
        with open('{}_compressed.pkl'.format(filename), 'wb') as f:
            pickle.dump(compressed_g, f)
