#!/usr/bin/env python

import math
import networkx as nx

from collections import defaultdict
from itertolls import combinations

# TODO: unit tests

def read_graph(filename):
    g = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            timestamp, source, target, weight = line.split()
            g[timestamp].add_edge(source, target, weight=weight)


def lambda_connection(graph, source, target, lambda_=5):
    # using shortest path weight as quality function
    path = nx.shortest_path(graph, source, target, cutoff=lambda_)
    edge_weights = nx.get_edge_attributes(g, 'weight')
    path_weight = sum(edge_weights[s, t] for s, t in zip(path, path[1:]))
    return 1/path_weight


def lambda_distance(graph1, graph2):
    # both graphs have to have the same vertex set
    assert fst_graph.nodes == snd_graph.nodes

    total = 0
    for source, target in zip(graph1.nodes, graph2.nodes):
        total += (lambda_connection(graph1, s, t) - lambda_connection(graph2, s, t))**2

    return math.sqrt(total)


def decompress(graph):
    # TODO: implement
    pass


def graph_merge(graph, source, target):
    # TODO: finish
    def new_weight(s, t):
        if s == t:
            return lambda_connection(graph, s, t)
    new_graph = graph.copy()
    # sum edge weights, including zero-edge weights


def create_compressed_graph(graph):
    # TODO: check this is valid
    new_graph = graph.copy()
    attributes = {'contains': node for node in graph.nodes}
    nx.set_node_attributes(new_graph, attributes)


def brute_force_greedy(graph, compression_ratio=0.1):
    compressed_graph = graph.copy()
    curr_compression_ratio = 0
    while curr_compression_ratio < compression_ratio:
        min_source, min_target = 0, 0
        min_distance = float('inf')
        for source, target in combinations(compressed_graph.nodes, 2):
            distance = lambda_distance(graph, decompress(compressed_graph))
            if distance < min_distance:
                min_source = source
                min_target = target
        compressed_graph = merge(compressed_graph, source, target)
        curr_compression_ratio = get_compression_ratio(graph, compressed_graph)

    return compressed_graph
