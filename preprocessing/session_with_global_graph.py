import networkx as nx
from preprocessing.session_process import SessionProcess
from data.config import data_read_path
import os
from itertools import chain
import pickle

# 类似EGES，将多个session合并到一张图中。
class SessionGlobalGraph():
    def __init__(self, read_path):
        with open(read_path, 'rb') as f:
            train_sess, test_sess, self.item2idx = pickle.load(f)
        self.G = self.construct_graph(train_sess, test_sess)

    def construct_graph(self, train_sess, test_sess):
        G = nx.Graph()
        for sess in train_sess + test_sess:
            for i in range(len(sess) - 1):
                if not G.has_edge(sess[i], sess[i+1]):
                    G.add_edge(sess[i], sess[i + 1], weight=1)
                else:
                    G[sess[i]][sess[i + 1]]['weight'] += 1
        for node in G.nodes():
            if node in test_sess:
                G.nodes[node]['train'] = False
            else:
                G.nodes[node]['train'] = True
        return G