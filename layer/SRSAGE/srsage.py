import tensorflow as tf
from layer.SRSAGE.ggnn import GGNN
from metrics.precision_mrr import PRE_MRR
import math
from layer.SRSAGE.graphsage import GraphSAGE

class SRSAGE(object):
    def __init__(self, hidden_size, learning_rate, l2_weight, topk, log, sage: GraphSAGE, ggnn: GGNN):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.l2_weight = l2_weight
        self.topk = topk
        self.log = log

        self.ggnn = ggnn
        self.sage = sage

        self.pre_mrr = PRE_MRR(topk=topk)

        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.in_adj = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        self.out_adj = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        self.item_sess = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.input_index = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, ])

        self.build()

    def build(self):
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

        self.loss, self.batch_pre, self.batch_mrr = self.call()
        if self.log is True:
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('P20', self.batch_pre)
            tf.summary.scalar('MRR20', self.batch_mrr)
        self.merge_ops = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def call(self, trainable=True):
        batch_item_embedding = self.ggnn.call(self.item_sess, in_adj=self.in_adj,
                                              out_adj=self.out_adj)  # [batch_size, max_node, hidden_size]
        with tf.name_scope('get_last_item_embedding'):
            last_index_indices = tf.reshape(tf.reduce_sum(self.mask, axis=1) - 1, [-1, 1])
            last_index = tf.gather(self.input_index, last_index_indices, batch_dims=-1) # [batch_size, 1]
            last_item_embedding = tf.gather(batch_item_embedding, last_index, batch_dims=-1) # [batch_size, 1, hidden_size]

        with tf.name_scope('get_input_item_embedding'):
            input_embedding = tf.gather(batch_item_embedding, self.input_index, batch_dims=-1)  # [batch_size, maxlen, hidden_size]

        with tf.name_scope('get_sample_agg_tensor'):
            self.sage_last_embedding = self.sage.sage() # [batch_size, hidden_size]

        with tf.name_scope('attention'):
            q = tf.matmul(last_item_embedding, self.W1)
            k = tf.matmul(input_embedding, self.W2)
            q_k = tf.sigmoid(q + k + self.b)
            # item为0对应的embedding应该被mask掉
            alpha = tf.matmul(q_k, self.Q) * tf.expand_dims(tf.cast(self.mask, tf.float32), axis=-1)  # [batch_size, maxlen, 1]
            graph_embedding = tf.reduce_sum(alpha * input_embedding, axis=1) # [batch_size, hidden_size]

        with tf.name_scope('add_tensor'):
            graph_embedding = graph_embedding + self.sage_last_embedding

        with tf.name_scope('get_candidate_item_score'):
            hybrid_embedding = tf.matmul(tf.concat([tf.squeeze(last_item_embedding), graph_embedding], axis=-1), self.W3)
            candidate_item_embedding = self.ggnn.embedding[1:]  # [从1开始的item, hidden_size]
            candidate_item_score = tf.matmul(hybrid_embedding, candidate_item_embedding, transpose_b=True)

        with tf.name_scope('metrics'):
            batch_pre, batch_mrr = self.pre_mrr.result(self.label - 1, candidate_item_score)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label - 1, logits=candidate_item_score))
            if trainable is False:
                return loss, batch_pre, batch_mrr
            l2_variables = [var for var in tf.trainable_variables() if len(var.shape) > 1]
            l2_loss = self.l2_weight * tf.reduce_sum([tf.nn.l2_loss(var) for var in l2_variables])
            return loss + l2_loss, batch_pre, batch_mrr