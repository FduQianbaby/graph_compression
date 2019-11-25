#!/usr/bin/env python3

import compression

import networkx as nx
import os
import tempfile
import unittest


def test_graph():
    g = nx.DiGraph()
    g.add_weighted_edges_from([('1', '2', 1), ('2', '3', 1), ('3', '2', 1), ('3', '4', 1), ('4', '5', 1), ('3', '5', 1)])
    attributes = {node: {'contains': set([node])} for node in g.nodes}
    nx.set_node_attributes(g, attributes)
    return g


def test_graph_1():
    g = nx.DiGraph()
    g.add_weighted_edges_from([('1', '2', 1), ('2', '1', 1)])
    attributes = {node: {'contains': set([node])} for node in g.nodes}
    nx.set_node_attributes(g, attributes)
    return g

def test_compressed_graph():
    g = nx.DiGraph()
    g.add_weighted_edges_from([('1+2+3+4', '5', 1), ('5', '6', 1)])
    attributes = {node: {'contains': set([node])} for node in g.nodes}
    attributes['1+2+3+4'] = {'contains': set(['1', '2', '3', '4'])}
    nx.set_node_attributes(g, attributes)
    return g


class TestCompression(unittest.TestCase):
    def test_make_all_dirs(self):
        compression.make_all_dirs(['./test0{}'.format(x) for x in range(5)])

        for x in range(5):
            directory = './test0{}'.format(x)
            self.assertTrue(os.path.isdir(directory))
            os.rmdir(directory)


    def test_read_graph(self):
        _, filename = tempfile.mkstemp()
        with open(filename, 'w') as f:
            for source, target in test_graph().edges:
                f.write("{} {} {} {}".format(source, target, 1, 0) + "\n")
            for source, target in test_graph_1().edges:
                f.write("{} {} {} {}".format(source, target, 1, 1) + "\n")

        g = compression.read_graph(filename)
        current_graph = test_graph()
        for s, t in test_graph_1().edges:
            current_graph.add_edge(s, t)
        self.assertTrue(nx.is_isomorphic(g['0'], test_graph()))
        self.assertTrue(nx.is_isomorphic(g['1'], current_graph))

        os.remove(filename)


    def test_lambda_connection(self):
        g = test_graph()
        source = 1

        pr = nx.pagerank(g, personalization={source: 1}, tol=1e-03)

        for node in g.nodes:
            self.assertEqual(pr[node], compression.lambda_connection(g, source, node))


    def test_lambda_distance(self):
        g = test_graph()
        for s, t in g.edges:
            self.assertEqual(compression.lambda_distance(g, g, s, t), 0)


    def test_decompress(self):
        g = test_compressed_graph()
        g1 = compression.decompress(g)
        for supersource, supertarget in g.edges:
            for source in supersource.split('+'):
                for target in supertarget.split('+'):
                    self.assertTrue(source in g1.nodes)
                    self.assertTrue(target in g1.nodes)
                    self.assertTrue((source, target) in g1.edges)


    def test_is_edge(self):
        g = test_graph()
        for edge in g.edges:
            self.assertTrue(compression.is_edge(g, edge))
        self.assertFalse(compression.is_edge(g, ('a', 'b')))


    def test_graph_merge(self):
        g = test_graph()
        g = compression.init_compressed_graph(g)
        g1 = compression.graph_merge(g, '1', '2')
        self.assertTrue('1+2' in g1.nodes)
        for node in g.neighbors('1'):
            if node == '2':
                continue
            self.assertTrue(('1+2', node) in g1.edges)

        for node in g.neighbors('2'):
            if node == '1':
                continue
            self.assertTrue(('1+2', node) in g1.edges)


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
        self.assertEqual(compression.calc_cr(g0, g1), len(g1.nodes)/len(g0.nodes))


    def test_two_hop_pairs(self):
        g = test_graph()
        nodes = list(compression.two_hop_pairs(g, '2'))

        nodes_in = ['3', '4', '5']
        nodes_out = ['1']

        for node in nodes_in:
            self.assertTrue(node in nodes)

        for node in nodes_out:
            self.assertFalse(node in nodes)


if __name__ == '__main__':
    unittest.main()
