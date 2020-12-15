from preprocessing.one_graph_process import OneGraphProcess
from layer.GraphSAGE.minibatch import EdgeMiniBatch
from layer.GraphSAGE.graphsage import GraphSAGE
from main.utils import generate_batch_index
import numpy as np
import tensorflow as tf
from collections import defaultdict
from layer.GraphSAGE.model.neigh_samplers import UniformNeighborSampler
from layer.GraphSAGE.model.models import SAGEInfo
from layer.GraphSAGE.model.models import SampleAndAggregate

np.random.seed(0)
tf.random.set_random_seed(0)


class LinkPredictionMain():
    def __init__(self):
        self.one_graph = OneGraphProcess()
        self.minibatch = EdgeMiniBatch(G=self.one_graph.G, item2idx=self.one_graph.item2idx, neighbor_num=20)
        self.graphsage = GraphSAGE(adj_table=self.minibatch.train_adj, nodes_degree=self.minibatch.degree,
                                   embedding_dim=8, negative_samples=19,
                                   neighbor_sample_list=[5, 10], weight_dim_list=[8, 8],
                                   concat_list=[True, True], learning_rate=1e-3)
        self.metrics = defaultdict(list)

        self.build_on_other_model()
        self.train_on_other_model(batch_size=128, epoch=100)

        # self.train(batch_size=128, epoch=100)

    def build_on_other_model(self):
        self.adj_info = tf.Variable(self.minibatch.train_adj, trainable=False, name="adj_info")
        sampler = UniformNeighborSampler(self.adj_info)
        layer_infos = [SAGEInfo("node", sampler, 10, 8),
                       SAGEInfo("node", sampler, 5, 8)]
        self.placeholders = {
            'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
            'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
            # negative samples for all nodes in the batch
            'neg_samples': tf.placeholder(tf.int32, shape=(None,),
                                          name='neg_sample_size'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),
        }
        self.model = SampleAndAggregate(self.placeholders,
                                        None,
                                        self.minibatch.train_adj,
                                        self.minibatch.degree,
                                        layer_infos=layer_infos,
                                        identity_dim=8, lr=1e-3, neg_sample=19, weight_decay=0,
                                        logging=False)

    def train_on_other_model(self, batch_size, epoch):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        train_edges = self.minibatch.train_edge
        for i in range(epoch):
            loss_epoch, mrr_epoch = 0, 0
            batch_index_list = generate_batch_index(len(train_edges), batch_size=batch_size, seed=i)
            for batch_idx in batch_index_list:
                batch_u_nodes, batch_v_nodes = [edge[0] for edge in train_edges[batch_idx]], \
                                               [edge[1] for edge in train_edges[batch_idx]]
                feed_dict = {self.placeholders['batch1']: batch_u_nodes, self.placeholders['batch2']: batch_v_nodes,
                             self.placeholders['dropout']: 0, self.placeholders['batch_size']: len(batch_idx)}
                batch_loss, batch_agg, batch_mrr, _ = self.sess.run([self.model.loss, self.model.outputs1,
                                                                     self.model.mrr, self.model.opt_op], feed_dict)
                loss_epoch += batch_loss * len(batch_idx)
                mrr_epoch += batch_mrr * len(batch_idx)
            self.metrics['train_loss'].append(loss_epoch / len(train_edges))
            self.metrics['train_mrr'].append(mrr_epoch / len(train_edges))
            print('================epoch : {}================'.format(i))
            print('train_loss : {}, MRR@20 : {}'.format(self.metrics['train_loss'][-1], self.metrics['train_mrr'][-1]))
            self.test_on_other_model(batch_size)

    def test_on_other_model(self, batch_size):
        self.sess.run(tf.assign(self.adj_info, self.minibatch.test_adj))
        loss_epoch, mrr_epoch = 0, 0
        test_edges = self.minibatch.test_edge
        batch_index_list = generate_batch_index(len(test_edges), batch_size=batch_size, seed=0)
        for batch_idx in batch_index_list:
            batch_u_nodes, batch_v_nodes = [edge[0] for edge in test_edges[batch_idx]], \
                                           [edge[1] for edge in test_edges[batch_idx]]
            feed_dict = {self.placeholders['batch1']: batch_u_nodes, self.placeholders['batch2']: batch_v_nodes,
                         self.placeholders['dropout']: 0, self.placeholders['batch_size']: len(batch_idx)}
            batch_loss, batch_mrr = self.sess.run([self.model.loss, self.model.mrr], feed_dict)
            loss_epoch += batch_loss * len(batch_idx)
            mrr_epoch += batch_mrr * len(batch_idx)
        self.metrics['test_loss'].append(loss_epoch / len(test_edges))
        self.metrics['test_mrr'].append(mrr_epoch / len(test_edges))
        print('test_loss : {}, MRR@20 : {}'.format(self.metrics['test_loss'][-1], self.metrics['test_mrr'][-1]))
        self.sess.run(tf.assign(self.adj_info, self.minibatch.train_adj))

    def train(self, batch_size, epoch):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        train_edges = self.minibatch.train_edge
        for i in range(epoch):
            loss_epoch, mrr_epoch = 0, 0
            batch_index_list = generate_batch_index(len(train_edges), batch_size=batch_size, seed=i)
            for batch_idx in batch_index_list:
                batch_u_nodes, batch_v_nodes = [edge[0] for edge in train_edges[batch_idx]], \
                                               [edge[1] for edge in train_edges[batch_idx]]
                feed_dict = {self.graphsage.u_nodes: batch_u_nodes, self.graphsage.v_nodes: batch_v_nodes}
                batch_loss, batch_mrr, _ = self.sess.run([self.graphsage.loss, self.graphsage.mrr,
                                                          self.graphsage.clip_grad_op],
                                                         feed_dict)
                loss_epoch += batch_loss * len(batch_idx)
                mrr_epoch += batch_mrr * len(batch_idx)
            self.metrics['train_loss'].append(loss_epoch / len(train_edges))
            self.metrics['train_mrr'].append(mrr_epoch / len(train_edges))
            print('================epoch : {}================'.format(i))
            print('train_loss : {}, MRR@20 : {}'.format(self.metrics['train_loss'][-1], self.metrics['train_mrr'][-1]))
            self.test(batch_size)

    def test(self, batch_size):
        self.sess.run(tf.assign(self.graphsage.adj_table, self.minibatch.test_adj))
        loss_epoch, mrr_epoch = 0, 0
        test_edges = self.minibatch.test_edge
        batch_index_list = generate_batch_index(len(test_edges), batch_size=batch_size, seed=0)
        for batch_idx in batch_index_list:
            batch_u_nodes, batch_v_nodes = [edge[0] for edge in test_edges[batch_idx]], \
                                           [edge[1] for edge in test_edges[batch_idx]]
            feed_dict = {self.graphsage.u_nodes: batch_u_nodes, self.graphsage.v_nodes: batch_v_nodes}
            batch_loss, mrr = self.sess.run([self.graphsage.loss, self.graphsage.mrr], feed_dict)
            loss_epoch += batch_loss * len(batch_idx)
            mrr_epoch += mrr * len(batch_idx)
        self.metrics['test_loss'].append(loss_epoch / len(test_edges))
        self.metrics['test_mrr'].append(mrr_epoch / len(test_edges))
        print('test_loss : {}, MRR@20 : {}'.format(self.metrics['test_loss'][-1], self.metrics['test_mrr'][-1]))
        self.sess.run(tf.assign(self.graphsage.adj_table, self.minibatch.train_adj))


if __name__ == '__main__':
    lp = LinkPredictionMain()
