import tensorflow as tf
from layer.base_layer import Layer

class GCN(Layer):
    def __init__(self, emb_dim):
        super(GCN, self).__init__()
        self.emb_dim = emb_dim

    def build(self):
        self.embedding = tf.get_variable(name='embedding', shape=[None, self.emb_dim], dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(0, 1))

    def call(self, inputs):
