import numpy as np
import tensorflow as tf

class EdgeMiniBatch():
    def __init__(self, G, item2idx, neighbor_num):
        self.G = G
        self.item2idx = item2idx
        self.neighbor_num = neighbor_num
        self.train_adj, self.degree = self.construct_train_adj() # 训练集对应的图没有测试集的节点以及对应的边
        self.test_adj = self.construct_all_adj()
        # self.train_edge, self.test_edge = self.get_train_test_edges()

    def construct_train_adj(self):
        # 无需编码
        adj = np.zeros([len(self.item2idx) + 1, self.neighbor_num], dtype=np.int32)
        degree = np.zeros([len(self.item2idx) + 1, ], dtype=np.int32)
        for u in self.G.nodes():
            if self.G.nodes[u]['train'] is False:
                continue
            train_neighbor = [v for v in self.G.neighbors(u) if self.G.nodes[v]['train'] is True]
            degree[u] = len(train_neighbor)
        for u in self.G.nodes():
            v_s = [v for v in self.G.neighbors(u)]
            total_degree = sum([degree[v] for v in v_s])
            norm_degree = [degree[v] / total_degree for v in v_s]
            if len(v_s) >= self.neighbor_num:
                adj[u] = np.random.choice(v_s, [self.neighbor_num, ], replace=False, p=norm_degree)
            else:
                adj[u] = np.random.choice(v_s, [self.neighbor_num, ], replace=True, p=norm_degree)
        return adj, degree

    def construct_all_adj(self):
        adj = np.zeros([len(self.item2idx) + 1, self.neighbor_num], dtype=np.int32)
        degree = np.zeros([len(self.item2idx) + 1, ], dtype=np.int32)
        for u in self.G.nodes():
            train_neighbor = [v for v in self.G.neighbors(u)]
            degree[u] = len(train_neighbor)
        for u in self.G.nodes():
            v_s = [v for v in self.G.neighbors(u)]
            total_degree = sum([degree[v] for v in v_s])
            norm_degree = [degree[v] / total_degree for v in v_s]
            if len(v_s) >= self.neighbor_num:
                adj[u] = np.random.choice(v_s, [self.neighbor_num, ], replace=False, p=norm_degree)
            else:
                adj[u] = np.random.choice(v_s, [self.neighbor_num, ], replace=True, p=norm_degree)
        return adj

    def get_train_test_edges(self):
        train_edges, test_edges = [], []
        for u, v in self.G.edges():
            if self.G.nodes[u]['train'] is False or self.G.nodes[v]['train'] is False:
                test_edges.append([self.item2idx[u], self.item2idx[v]])
            else:
                train_edges.append([self.item2idx[u], self.item2idx[v]])
        return np.asarray(train_edges), np.asarray(test_edges)

