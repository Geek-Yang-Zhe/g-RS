import networkx as nx
from preprocessing.base_process import BaseProcess
from data.config import data_read_path
import os
from itertools import chain


# 类似EGES，将多个session合并到一张图中。
class OneGraphProcess(BaseProcess):
    def __init__(self):
        self.read_path = os.path.join(data_read_path, 'diginetica_sample.csv')
        self.sess_dict, self.date_dict = self.generate_sess_date_dict(self.read_path)
        self.G = self.construct_graph(self.sess_dict.values()) # 有些session为长度为1，在这里不会构成边
        self.encoded_item_list, self.item2idx = self.encode_sess([self.G.nodes], merge_sess=True)
        self.train_test_node_edge_split(day=7)

    def construct_graph(self, sess_list):
        G = nx.DiGraph()
        for sess in sess_list:
            for i in range(len(sess) - 1):
                if not G.has_edge(sess[i], sess[i+1]):
                    G.add_edge(sess[i], sess[i + 1], weight=1)
                else:
                    G[sess[i]][sess[i + 1]]['weight'] += 1
        return G

    def train_test_node_edge_split(self, day):
        G = self.G
        _, test_sess_list = self.train_test_split(self.sess_dict, self.date_dict, day=day)
        test_node = set(chain.from_iterable(test_sess_list))
        for node in G.nodes():
            if node in test_node:
                G.nodes[node]['train'] = False
            else:
                G.nodes[node]['train'] = True
        train_edge_num = 0
        for u, v in G.edges():
            if G.nodes[u]['train'] is False or G.nodes[v]['train'] is False:
                G[u][v]['train'] = False
            else:
                G[u][v]['train'] = True
                train_edge_num += 1
        print('训练边数: {}, 测试边数: {}'.format(train_edge_num, len(G.edges) - train_edge_num))
        print('训练点数: {}, 测试点数: {}'.format(len(G.nodes) - len(test_node), len(test_node)))