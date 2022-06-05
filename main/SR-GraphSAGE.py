import sys
sys.path.append('../')
from main.utils import padding_mask, generate_batch_index, generate_batch_AdjMatrix
import pickle
from layer.SRSAGE.srsage import SRSAGE
import os
import tensorflow as tf
from collections import defaultdict
import csv
from layer.SRSAGE.ggnn import GGNN
from preprocessing.session_with_global_graph import SessionGlobalGraph
from layer.SRSAGE.minibatch import EdgeMiniBatch
from layer.SRSAGE.graphsage import GraphSAGE
import numpy as np

# 确保可以复现
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
tf.set_random_seed(0)

# diginetica 43105
class SessionMain(object):
    def __init__(self):
        self.one_graph = SessionGlobalGraph(read_path='../processed_data/diginetica/encoded_sess')
        train_data, train_label, test_data, test_label = self.get_inputs(path='../processed_data/diginetica')
        self.minibatch = EdgeMiniBatch(G=self.one_graph.G, item2idx=self.one_graph.item2idx, neighbor_num=20)
        self.train_mask, self.train_sess, self.train_label = padding_mask(train_data, train_label)
        self.test_mask, self.test_sess, self.test_label = padding_mask(test_data, test_label)
        self.metrics = defaultdict(list)
        self.ggnn = GGNN(hidden_size=8, steps=1, num_item=1 + len(self.one_graph.item2idx))
        self.graphsage = GraphSAGE(adj_table=self.minibatch.train_adj, nodes_degree=self.minibatch.degree,
                                   embedding_dim=8, negative_samples=19,
                                   neighbor_sample_list=[5, 10], weight_dim_list=[8, 8],
                                   concat_list=[True, False], learning_rate=1e-3)
        self.srgnn = SRSAGE(hidden_size=8, learning_rate=1e-3, l2_weight=1e-5, topk=20, log=True,
                            sage=self.graphsage, ggnn=self.ggnn)
        self.train_writer = tf.summary.FileWriter(logdir='../tensorboard/train', graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(logdir='../tensorboard/test')
        self.train(epoch=10, batch_size=100)
        self.train_writer.close()
        self.test_writer.close()
        self.save_metrics(path='../metrics/sample1.csv')

    def get_inputs(self, path):
        with open(os.path.join(path, 'train.pkl'), 'rb') as f:
            train_data, train_label = pickle.load(f)
            self.train_last = np.array([x[-1] for x in train_data])
        with open(os.path.join(path, 'test.pkl'), 'rb') as f:
            test_data, test_label = pickle.load(f)
            self.test_last = np.array([x[-1] for x in test_data])
        return train_data, train_label, test_data, test_label

    def get_feed_dict(self, batch_index, sess, label, mask, mode='train'):
        batch_sess = sess[batch_index]
        batch_label = label[batch_index]
        batch_mask = mask[batch_index]
        batch_out_adj, batch_in_adj, batch_padding_sess, batch_input_index = generate_batch_AdjMatrix(
            batch_sess)
        feed_dict = {self.srgnn.item_sess: batch_padding_sess, self.srgnn.label: batch_label,
                     self.srgnn.mask: batch_mask, self.srgnn.input_index: batch_input_index,
                     self.srgnn.in_adj: batch_in_adj, self.srgnn.out_adj: batch_out_adj}
        "加入graphsage需要的u节点"
        if mode == 'test':
            feed_dict[self.graphsage.u_nodes] = self.test_last[batch_index]
        else:
            feed_dict[self.graphsage.u_nodes] = self.train_last[batch_index]
        return feed_dict

    def train(self, epoch, batch_size):
        total_samples = len(self.train_sess)
        for i in range(epoch):
            loss_epoch, precision_epoch, mrr_epoch = 0, 0, 0
            batch_index_list = generate_batch_index(total_samples, batch_size=batch_size, seed=i)
            for batch_index in batch_index_list:
                feed_dict = self.get_feed_dict(batch_index, self.train_sess, self.train_label, self.train_mask)
                fetches = [self.srgnn.merge_ops, self.srgnn.global_step,
                           self.srgnn.loss, self.srgnn.batch_pre, self.srgnn.batch_mrr, self.srgnn.optimizer]
                merged_summary, step, batch_loss, batch_precision, batch_mrr, _ = \
                    self.srgnn.sess.run(fetches, feed_dict)
                # print(u_embedding)
                self.train_writer.add_summary(merged_summary, step)
                loss_epoch += batch_loss * len(batch_index)
                precision_epoch += batch_precision * len(batch_index)
                mrr_epoch += batch_mrr * len(batch_index)
            self.metrics['train_loss'].append(loss_epoch / total_samples)
            self.metrics['train_precision'].append(precision_epoch / total_samples)
            self.metrics['train_mrr'].append(mrr_epoch / total_samples)
            print('================epoch : {}================'.format(i))
            print('train_loss : {}, P@20 : {}, MRR@20 : {}'.format(self.metrics['train_loss'][-1],
                                                                   self.metrics['train_precision'][-1],
                                                                   self.metrics['train_mrr'][-1]))
            self.test(batch_size)

    # batch_size应尽可能大
    def test(self, batch_size):
        self.srgnn.sess.run(tf.assign(self.graphsage.adj_table, self.minibatch.test_adj))
        loss_epoch, precision_epoch, mrr_epoch = 0, 0, 0
        total_samples = len(self.test_sess)
        batch_index_list = generate_batch_index(total_samples, batch_size=batch_size, seed=0)
        for i, batch_index in enumerate(batch_index_list):
            feed_dict = self.get_feed_dict(batch_index, self.test_sess, self.test_label, self.test_mask,
                                           mode='test')
            fetches = [self.srgnn.merge_ops, self.srgnn.global_step,
                       self.srgnn.loss, self.srgnn.batch_pre, self.srgnn.batch_mrr]
            merged_summary, step, batch_loss, batch_precision, batch_mrr = self.srgnn.sess.run(fetches, feed_dict)
            loss_epoch += batch_loss * len(batch_index)
            precision_epoch += batch_precision * len(batch_index)
            mrr_epoch += batch_mrr * len(batch_index)
        # 只使用最后一个batch作为可视化
        self.test_writer.add_summary(merged_summary, step)
        self.metrics['test_loss'].append(loss_epoch / total_samples)
        self.metrics['test_precision'].append(precision_epoch / total_samples)
        self.metrics['test_mrr'].append(mrr_epoch / total_samples)
        print('test_loss : {}, P@20 : {}, MRR@20 : {}'.format(self.metrics['test_loss'][-1],
                                                               self.metrics['test_precision'][-1],
                                                               self.metrics['test_mrr'][-1]))

    def save_metrics(self, path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            head = ['epoch'] + list(self.metrics.keys())
            writer.writerow(head)
            for epoch in range(len(self.metrics['train_loss'])):
                row = [epoch] + [self.metrics[key][epoch] for key in self.metrics.keys()]
                writer.writerow(row)

if __name__ == '__main__':
    main = SessionMain()
