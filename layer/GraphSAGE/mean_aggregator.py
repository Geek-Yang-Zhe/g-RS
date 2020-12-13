import tensorflow as tf


class MeanAggregator():
    def __init__(self, input_dim, output_dim, layer_idx):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_id = layer_idx
        self.build()

    def build(self):
        with tf.variable_scope('gnn_layer_{}'.format(self.layer_id)):
            self.w_u = tf.get_variable(name='mean_agg_w_u', dtype=tf.float32,
                                       initializer=tf.random.uniform([self.input_dim, self.output_dim],
                                                                     -tf.sqrt(6.0) / self.input_dim,
                                                                     tf.sqrt(6.0) / self.input_dim))
            self.w_v = tf.get_variable(name='mean_agg_w_v', dtype=tf.float32,
                                       initializer=tf.random.uniform([self.input_dim, self.output_dim],
                                                                     -tf.sqrt(6.0) / self.input_dim,
                                                                     tf.sqrt(6.0) / self.input_dim))

    def __call__(self, inputs, *args, **kwargs):
        outputs =  self.call(inputs=inputs)
        return outputs

    def call(self, inputs):
        u_embedding, v_embedding = inputs  # [num_u, dim], [num_u, num_example, dim]
        mean_v_embedding = tf.matmul(tf.reduce_mean(v_embedding ,axis=1), self.w_v)
        mean_u_embedding = tf.matmul(u_embedding, self.w_u)
        outputs = tf.nn.relu(tf.add(mean_u_embedding, mean_v_embedding))
        return outputs