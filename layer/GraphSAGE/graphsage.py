import tensorflow as tf


class GraphSAGE():
    def __init__(self, adj_table, embedding_dim, negative_samples, neighbor_sample_list):
        self.adj_table = adj_table
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.neighbor_sample_list = neighbor_sample_list
        self.u_nodes = tf.placeholder(name='u_nodes', shape=[None, ], dtype=tf.int32)
        self.v_nodes = tf.placeholder(name='v_nodes', shape=[None, ], dtype=tf.int32)
        self.build()

    def build(self):
        self.embedding = tf.get_variable(name='embedding', shape=[self.adj_table.shape[0], self.embedding_dim],
                                         dtype=tf.float32, initializer=tf.random_normal_initializer(0, 1))
        self._build()

    def _build(self):
        self.sample_neighbor_idx = self.sample_from_neighbor(self.u_nodes, self.neighbor_sample_list[0])

    def sample_from_neighbor(self, u_ids, num_samples):
        neighbor_idx = tf.nn.embedding_lookup(self.adj_table, u_ids)
        sample_idx = tf.random.shuffle(tf.range(self.adj_table.shape[-1]))[: num_samples]
        sample_neighbor_idx = tf.gather(neighbor_idx, sample_idx, axis=1)
        return sample_neighbor_idx

    # def sample(self, input_nodes):
