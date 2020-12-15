from model.utils import padding_mask, generate_batch_index, generate_batch_AdjMatrix
import pickle
from layer.SRGNN.SRGNN import SRGNNLayer
import os
import tensorflow as tf
from collections import defaultdict
import csv
from data.config import data_save_path

# 确保可以复现
tf.set_random_seed(0)

# diginetica 43105
class SessionMain(object):
    def __init__(self):
        self.train_data, self.train_label, self.test_data, self.test_label = self.get_inputs(
            path=os.path.join(data_save_path, 'diginetica_sample_1'))
        self.train_mask, self.train_sess, self.train_label = padding_mask(self.train_data, self.train_label)
        self.test_mask, self.test_sess, self.test_label = padding_mask(self.test_data, self.test_label)
        self.metrics = defaultdict(list)
        self.srgnn = SRGNNLayer(num_item=312, hidden_size=100, learning_rate=1e-3,
                                l2_weight=1e-5, ggnn_step=1, topk=20, log=True)
        self.train_writer = tf.summary.FileWriter(logdir='../tensorboard/train', graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(logdir='../tensorboard/test')
        self.train(epoch=10, batch_size=100)
        self.train_writer.close()
        self.test_writer.close()
        self.save_metrics(path='../metrics/sample1.csv')

    def get_inputs(self, path):
        with open(os.path.join(path, 'train.pkl'), 'rb') as f:
            train_data, train_label = pickle.load(f)
        with open(os.path.join(path, 'test.pkl'), 'rb') as f:
            test_data, test_label = pickle.load(f)
        return train_data, train_label, test_data, test_label

    def get_feed_dict(self, batch_index, sess, label, mask):
        batch_sess = sess[batch_index]
        batch_label = label[batch_index]
        batch_mask = mask[batch_index]
        batch_out_adj, batch_in_adj, batch_padding_sess, batch_input_index = generate_batch_AdjMatrix(
            batch_sess)
        feed_dict = {self.srgnn.item_sess: batch_padding_sess, self.srgnn.label: batch_label,
                     self.srgnn.mask: batch_mask, self.srgnn.input_index: batch_input_index,
                     self.srgnn.in_adj: batch_in_adj, self.srgnn.out_adj: batch_out_adj}
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
                merged_summary, step, batch_loss, batch_precision, batch_mrr, _ = self.srgnn.sess.run(fetches, feed_dict)
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
        loss_epoch, precision_epoch, mrr_epoch = 0, 0, 0
        total_samples = len(self.test_sess)
        batch_index_list = generate_batch_index(total_samples, batch_size=batch_size, seed=0)
        for i, batch_index in enumerate(batch_index_list):
            feed_dict = self.get_feed_dict(batch_index, self.test_sess, self.test_label, self.test_mask)
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
