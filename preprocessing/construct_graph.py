import networkx as nx
from preprocessing.base_process import BaseProcess
from data.config import data_read_path
import os

# 类似EGES，将多个session合并到一张图中。
class ConstuctGraph(BaseProcess):
    def __init__(self):
        self.read_path = os.path.join(data_read_path, 'diginetica_sample.csv')
        self.G = self.construct_graph()

    def construct_graph(self):
        sess_dict, _ = self.generate_sess_date_dict(self.read_path)
        encoded_sess_list, _ = self.encode_item(sess_dict.values())
        G = nx.DiGraph()
        for sess in encoded_sess_list:
            for i in range(len(sess) - 1):
                if not G.has_edge(sess[i], sess[i+1]):
                    G.add_edge(sess[i], sess[i + 1], weight=1)
                else:
                    G[sess[i]][sess[i + 1]]['weight'] += 1
        return G