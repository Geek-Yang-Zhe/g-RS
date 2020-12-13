import tensorflow as tf


class MeanAggregator():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self):
        self.w = tf.get_variable(name='mean_agg_w', dtype=tf.float32,
                                 initializer=tf.random.uniform(-tf.sqrt(6) / self.input_dim,
                                                               tf.sqrt(6) / self.input_dim))

    def __call__(self, inputs, *args, **kwargs):
        self.call(inputs=inputs)

    def call(self, inputs):


