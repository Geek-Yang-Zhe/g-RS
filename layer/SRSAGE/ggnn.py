import tensorflow as tf
import math
import numpy as np

class GGNN():
    def __init__(self, hidden_size, steps, num_item):
        self.num_item = num_item
        self.hidden_size = hidden_size
        self.steps = steps

        self.build()

    def build(self):
        # 这里第一个为补零的embedding，其他为正常的item
        def init_func(shape, dtype, partition_info):
            np.random.seed(0)
            initializer_val = np.random.rand(1 + self.num_item, self.hidden_size)
            initializer_val[0, :] = 0.0
            return initializer_val

        with tf.variable_scope("embedding"):
            self.embedding = tf.get_variable(name='embedding', shape=[1 + self.num_item, self.hidden_size],
                                             dtype=tf.float32,
                                             initializer=init_func)
        self.W_in = tf.get_variable(name='W_in', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                              math.sqrt(1 / self.hidden_size)))
        self.b_in = tf.get_variable(name='bias_in', shape=[self.hidden_size, ], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                              math.sqrt(1 / self.hidden_size)))
        self.W_out = tf.get_variable(name='W_out', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                               math.sqrt(1 / self.hidden_size)))
        self.b_out = tf.get_variable(name='bias_out', shape=[self.hidden_size, ], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                               math.sqrt(1 / self.hidden_size)))
        self.W_gate = tf.get_variable(name='W_gate_r_u', shape=[3 * self.hidden_size, 2 * self.hidden_size],
                                      dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                                math.sqrt(1 / self.hidden_size)))
        self.W_candidate = tf.get_variable(name='W_candidate', shape=[3 * self.hidden_size, self.hidden_size],
                                           dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-math.sqrt(1 / self.hidden_size),
                                                                                     math.sqrt(1 / self.hidden_size)))

    def call(self, sess, out_adj, in_adj):
        last_state = tf.nn.embedding_lookup(self.embedding, sess) #[batch_size, max_node, emb_dim]
        # 只是借用了GRU_Cell的方法，并不是真正的RNN
        for step in range(self.steps):
            with tf.variable_scope('ggnn'):
                state_in = tf.matmul(last_state, self.W_in) + self.b_in
                state_out = tf.matmul(last_state, self.W_out) + self.b_out
                rnn_input = tf.concat([tf.matmul(out_adj, state_out), tf.matmul(in_adj, state_in)], axis=-1)
                with tf.variable_scope('gru'):
                    gate = tf.matmul(tf.concat([rnn_input, last_state], axis=-1), self.W_gate)
                    sigmoid_gate = tf.sigmoid(gate)
                    update_gate, reset_gate = tf.split(sigmoid_gate, 2, axis=2) # [batch_size, max_node, hid_dim]
                    reset_state = reset_gate * last_state
                    candidate_state = tf.tanh(tf.matmul(tf.concat([rnn_input, reset_state], axis=-1), self.W_candidate))
                    last_state = (1 - update_gate) * last_state + update_gate * candidate_state
        return last_state
