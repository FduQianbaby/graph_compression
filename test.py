#!/usr/bin/env python3

import compression

import networkx as nx
import os
import tempfile
import unittest


def test_graph():
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (2, 3), (3, 2), (3, 4), (4, 5), (3, 5)])
    return g


def test_graph_1():
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (2, 1)])
    return g

def test_compressed_graph():
    g = nx.DiGraph()
    g.add_edges_from([('1+2+3+4', 5), (5, 6)])
    return g


class TestCompression(unittest.testcase):
    def test_make_all_dirs(self):
        compression.make_all_dirs(['./test0{}'.format(x) for x in range(5)])

        for x in range(5):
            directory = './test0{}'.format(x)
            self.assertTrue(os.path.isdir(directory))
            os.remove(directory)


    def test_read_graph(self):
        f, filename = tempfile.mkstemp()
        for source, target in test_graph().edges:
            f.write(" ".join(source, target, 1, 0 + '\n'))
        for source, target in test_graph_1().edges:
            f.write(" ".join(source, target, 1, 0 + '\n'))
        f.close()

        g = read_graph(filename)

        self.assertTrue(nx.is_isomorphic(g[0], test_graph()))
        self.assertTrue(nx.is_isomorphic(g[1], test_graph_1()))

        os.remove(filename)


    def test_lambda_connection(self):
        g = test_graph()
        source = g.nodes[0]

        pr = nx.pagerank(g, personalization={source: 1})

        for node in g.nodes:
            self.assertEqual(pr[node], compression.lambda_connection(g, source, node))


    def test_lambda_distance(self):
        g = test_graph()
        self.assertEqual(compression.lambda_distance(g, g), 0)
        # TODO: add other test


    def test_decompress(self):
        g = test_compressed_graph()
        g1 = compression.decompress(g)
        for supersource, supertarget in g.edges:
            for source in supersource.split('+'):
                for target in supertarget.split('+'):
                    self.assertTrue(source in g1)
                    self.assertTrue(target in g1)
                    self.assertTrue((source, target) in g1)


    def test_is_edge(self):
        g = test_graph()
        for edge in g.edges:
            self.assertTrue(compression.is_edge(g, edge))
        self.assertFalse(compression.is_edge(g, ('a', 'b')))


    def test_graph_merge(self):
        g = test_graph()
        g1 = compression.merge(g, 1, 2)
        self.assertTrue('1+2' in g1.nodes)
        for node in g.neighbors(1) + g.neighbors(2):
            self.assertTrue(('1+2', node) in g1)


    def test_init_compressed_graph(self):
        g0 = test_graph()
        g1 = compression.init_compressed_graph(g0)
        contains = nx.get_node_attributes(g1, 'contains')

        for node in g0.nodes:
            self.assertTrue(node in g1.nodes)
            self.assertEqual(contains[node], set([node]))


    def test_calc_cr(self):
        g0 = test_graph()
        g1 = test_graph_1()

        self.assertEqual(compression.calc_cr(g0, g0), 1)
        self.assertEqual(compression.calc_cr(g1, g1), 1)
        self.assertEqual(compression.calc_cr(g0, g1), len(g0.nodes)/len(g1.nodes))


    def test_all_two_hop_pairs(self):
        g = test_graph()
        pairs = compression.all_two_hop_pairs(g)

        pairs_in = [(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
        pairs_out = [(1, 5), (1, 4), (3, 1)]

        for pair in pairs_in:
            self.assertTrue(pair in pairs)

        for pair in pairs_out:
            self.assertFalse(pair in pairs)


    def test_brute_force_greedy(self):
        pass
