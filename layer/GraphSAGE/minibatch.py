import numpy as np
import tensorflow as tf

class EdgeMiniBatch():
    def __init__(self, G, item2idx, neighbor_num):
        self.G = G
        self.item2idx = item2idx
        self.neighbor_num = neighbor_num
        self.train_adj, self.degree = self.construct_train_adj() # 训练集对应的图没有测试集的节点以及对应的边
        self.test_adj = self.construct_all_adj()
        self.train_edge, self.test_edge = self.get_train_test_edges()

    def construct_train_adj(self):
        # idx从1开始编码
        adj = np.zeros([len(self.item2idx) + 1, self.neighbor_num], dtype=np.int32)
        degree = np.zeros([len(self.item2idx) + 1, ], dtype=np.int32)
        for u in self.G.nodes():
            if self.G.nodes[u]['train'] is False:
                continue
            train_neighbor = [self.item2idx[v] for v in self.G.neighbors(u) if self.G.nodes[v]['train'] is True]
            degree[self.item2idx[u]] = len(train_neighbor)
            if len(train_neighbor) != 0:
                if len(train_neighbor) >= self.neighbor_num:
                    adj[self.item2idx[u]] = np.random.choice(train_neighbor, [self.neighbor_num, ], replace=False)
                else:
                    adj[self.item2idx[u]] = np.random.choice(train_neighbor, [self.neighbor_num, ], replace=True)
        return adj, degree

    def construct_all_adj(self):
        adj = np.zeros([len(self.item2idx) + 1, self.neighbor_num], dtype=np.int32)
        for u in self.G.nodes():
            train_neighbor = [self.item2idx[v] for v in self.G.neighbors(u)]
            if len(train_neighbor) != 0:
                if len(train_neighbor) >= self.neighbor_num:
                    adj[self.item2idx[u]] = np.random.choice(train_neighbor, [self.neighbor_num, ], replace=False)
                else:
                    adj[self.item2idx[u]] = np.random.choice(train_neighbor, [self.neighbor_num, ], replace=True)
        return adj

    def get_train_test_edges(self):
        train_edges, test_edges = [], []
        for u, v in self.G.edges():
            if self.G.nodes[u]['train'] is False or self.G.nodes[v]['train'] is False:
                test_edges.append([self.item2idx[u], self.item2idx[v]])
            else:
                train_edges.append([self.item2idx[u], self.item2idx[v]])
        return np.asarray(train_edges), np.asarray(test_edges)

