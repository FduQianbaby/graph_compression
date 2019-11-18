#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import sys
from compression import read_graph, make_all_dirs


def usage(code):
    print('Usage: draw_graph.py [filename]')
    exit(code)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage(1)

    filename = sys.argv[1].strip()
    make_all_dirs(['draw_graph_files'])
    prefix = filename.split('/')[-1].split('.')[0]

    g = read_graph(filename)
    for timestamp, graph in g.items():
        nx.draw(graph)
        plt.savefig('draw_graph_files/{}_{}.png'.format(prefix, timestamp))
