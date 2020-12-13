from preprocessing.one_graph_process import OneGraphProcess
from layer.GraphSAGE.minibatch import EdgeMiniBatch
from layer.GraphSAGE.graphsage import GraphSAGE
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_random_seed(0)

class LinkPredictionMain():
    def __init__(self):
        self.one_graph = OneGraphProcess()
        self.minibatch = EdgeMiniBatch(G=self.one_graph.G, item2idx=self.one_graph.item2idx, neighbor_num=20,
                                       batch_size=64)
        self.graphsage = GraphSAGE(adj_table=self.minibatch.train_adj, embedding_dim=8, negative_samples=5,
                                   neighbor_sample_list=[8, 4, 2], weight_dim_list=[32, 16, 8])
        self.train()

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        feed_dict = {self.graphsage.u_nodes: [0, 1, 9],
                     self.graphsage.v_nodes: [3, 4, 5]}

        res = sess.run([self.graphsage.u_embedding, self.graphsage.v_embedding], feed_dict)
        for node_embedding in res:
            print(node_embedding)
        # print(self.minibatch.train_adj[[0, 1, 2], :])


if __name__ == '__main__':
    lp = LinkPredictionMain()