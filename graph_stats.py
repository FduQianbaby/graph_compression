#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pickle
import sys
from compression import make_all_dirs


STATS = ['number of edges', 'number of nodes', 'L1 norm of first eigenvector', 'L2 norm of first eigenvector']


def usage(code):
    print('Usage: graph_stats.py [filename]')
    exit(code)

def L1norm(vector):
    total = 0
    for elem in vector:
        total += abs(elem)

    return float(total)


def log(lst):
    # dont want generator, b/c matplotlib can't handle it
    return list([math.log10(x) if x != 0 else 0 for x in lst])


def plot(g, stats, filename):
    for label, stat in stats.items():
        if label == 'timestamps':
            continue
        plt.plot(stats['timestamps'], stat, label=label)

    plt.legend()
    plt.title('Metrics over time on compressed graph')
    plt.xlabel('Year')
    plt.ylabel('log10(Metric value)')
    plt.savefig('./plots/' + filename + '.png')
    plt.show()


def pad_vector(v, length):
    return numpy.pad(v, [[0, length], [0, 0]], 'constant')


def calc_stats(g, take_log=True):
    stats = {stat: [] for stat in STATS}
    stats['timestamps'] = list([t for t in g])
    last_eig = numpy.zeros(shape=(2, 1))

    for timetamp, graph in g.items():
        stats['number of nodes'].append(len(graph.nodes))
        stats['number of edges'].append(len(graph.edges))

        # eigenvalues
        vals, vectors = numpy.linalg.eig(nx.convert_matrix.to_numpy_matrix(graph))
        # pad eigenvectors with zeros, so we can compare
        new_len = vectors[:,1].shape[0]
        curr_len = last_eig.shape[0]
        curr_eig = vectors[:,1]

        curr_eig = pad_vector(curr_eig, curr_len - new_len) if curr_len > new_len else curr_eig
        last_eig = pad_vector(last_eig, new_len - curr_len) if new_len > curr_len else last_eig

        change_vec = numpy.subtract(curr_eig, last_eig)
        stats['L1 norm of first eigenvector'].append(L1norm(change_vec))
        stats['L2 norm of first eigenvector'].append(numpy.linalg.norm(change_vec))
        last_eig = vectors[:, 1]

    if take_log:
        stats['number of edges'] = log(stats['number of edges'])
        stats['number of nodes'] = log(stats['number of nodes'])
        stats['L1 norm of first eigenvector'] = log(stats['L1 norm of first eigenvector'])
        stats['L2 norm of first eigenvector'] = log(stats['L2 norm of first eigenvector'])

    return stats



if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage(1)

    filename = sys.argv[1].strip()
    prefix = filename.split('/')[-1].split('.')[0]
    make_all_dirs(['plots'])
    with open(filename, 'rb') as f:
        g = pickle.load(f)

    stats = calc_stats(g)
    plot(g, stats, prefix)
