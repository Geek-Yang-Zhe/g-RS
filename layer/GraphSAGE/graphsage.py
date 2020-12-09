import tensorflow as tf

class GraphSAGE():
    def __init__(self):
        self.u_nodes = tf.placeholder(name='u_nodes', shape=[None, ], dtype=tf.int32)
        self.v_nodes = tf.placeholder(name='v_nodes', shape=[None, ], dtype=tf.int32)

        self.build()

    def build(self):
        #

    def sample(self, input_nodes):
        # 对每个batch中每个样本的邻居节点采样
