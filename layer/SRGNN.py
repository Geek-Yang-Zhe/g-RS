import tensorflow as tf
from layer.ggnn import GGNN
import math
import sys


class SRGNNLayer(object):
    def __init__(self, maxlen, num_item, hidden_size, batch_size, learning_rate, l2_weight):
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = learning_rate
        self.l2_weight = l2_weight

        self.ggnn = GGNN(hidden_size=self.hidden_size, steps=1, num_item=num_item)

        self.mask = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.maxlen])
        self.in_adj = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.out_adj = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.item = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        self.input_index = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.maxlen])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, ])

        self.build()

    def build(self, mode='train'):
        self.W1 = tf.get_variable(name='w1_att', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                            math.sqrt(1 / self.hidden_size)))
        self.W2 = tf.get_variable(name='w2_att', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                            math.sqrt(1 / self.hidden_size)))
        self.b = tf.get_variable(name='att_bias', shape=[self.hidden_size, ], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                           math.sqrt(1 / self.hidden_size)))
        self.Q = tf.get_variable(name='att_Q', shape=[self.hidden_size, 1], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                           math.sqrt(1 / self.hidden_size)))
        self.W3 = tf.get_variable(name='w3_hybrid', shape=[2 * self.hidden_size, self.hidden_size], dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                            math.sqrt(1 / self.hidden_size)))
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        self.loss = self.call()
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def call(self, trainable=True):
        batch_item_embedding = self.ggnn.call(self.item, in_adj=self.in_adj,
                                              out_adj=self.out_adj)  # [batch_size, max_node, hidden_size]
        with tf.name_scope('get_last_item_embedding'):
            last_index_indices = tf.stack([tf.range(self.batch_size), tf.reduce_sum(self.mask, axis=1) - 1], axis=1)
            last_index = tf.gather_nd(self.input_index, last_index_indices)
            last_item_indices = tf.stack([tf.range(self.batch_size), last_index], axis=1)
            last_item_embedding = tf.gather_nd(batch_item_embedding, last_item_indices)

        with tf.name_scope('get_input_item_embedding'):
            input_batch = tf.tile(tf.expand_dims(tf.range(self.batch_size), 1), [1, self.maxlen])
            input_item_indices = tf.stack([input_batch, self.input_index], axis=-1)
            input_embedding = tf.gather_nd(batch_item_embedding,
                                           input_item_indices)  # [batch_size, maxlen, hidden_size]

        with tf.name_scope('attention'):
            q = tf.matmul(last_item_embedding, self.W1)
            k = tf.matmul(input_embedding, self.W2)
            q_k = tf.sigmoid(tf.reshape(q, [self.batch_size, 1, self.hidden_size]) + k + self.b)
            # item为0对应的embedding应该被mask掉
            alpha = tf.matmul(q_k, self.Q) * tf.expand_dims(tf.cast(self.mask, tf.float32),
                                                            axis=-1)  # [batch_size, maxlen, 1]
            graph_embedding = tf.reduce_sum(alpha * input_embedding, axis=1)

        with tf.name_scope('get_candidate_item_score'):
            hybrid_embedding = tf.matmul(tf.concat([last_item_embedding, graph_embedding], axis=-1), self.W3)
            candidate_item_embedding = self.ggnn.embedding[1:]  # [从1开始的item, hidden_size]
            candidate_item_score = tf.matmul(hybrid_embedding, candidate_item_embedding, transpose_b=True)

        with tf.name_scope('loss'):
            loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=candidate_item_score))
            if trainable is False:
                return loss
            l2_variables = [var for var in tf.trainable_variables() if len(var.shape) > 1]
            l2_loss = self.l2_weight * tf.reduce_sum([tf.nn.l2_loss(var) for var in l2_variables])
            return loss + l2_loss
