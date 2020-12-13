from layer.GraphSAGE.mean_aggregator import MeanAggregator
import tensorflow as tf
import math

class GraphSAGE():
    def __init__(self, adj_table, embedding_dim, negative_samples, neighbor_sample_list, weight_dim_list):
        self.adj_table = adj_table
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.neighbor_sample_list = neighbor_sample_list
        self.weight_dim_list = [self.embedding_dim] + weight_dim_list
        self.u_nodes = tf.placeholder(name='u_nodes', shape=[None, ], dtype=tf.int32)
        self.v_nodes = tf.placeholder(name='v_nodes', shape=[None, ], dtype=tf.int32)
        self.build()

    def build(self):
        self.embedding = tf.get_variable(name='embedding', shape=[self.adj_table.shape[0], self.embedding_dim],
                                         dtype=tf.float32)
        self.W = tf.get_variable(name='W1', shape=[2 * self.embedding_dim, self.embedding_dim], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(-math.sqrt(2 / self.embedding_dim),
                                                                           math.sqrt(2 / self.embedding_dim)))
        self._build()

    def _build(self):
        self.sample_idx_each_layer = self.sample_for_each_layer()
        self.u_embedding = self.aggregate(self.sample_idx_each_layer)

    def aggregate(self, sample_idx_each_layer):
        # 第0层聚合：1 -> 0, 2 -> 1, ..., total_layers -> total_layers - 1
        # 第1层聚合：1 -> 0, 2 -> 1, ..., total_layers - 1 -> total_layers - 2
        # 第total_layers - 1层聚合: 1 -> 0
        layer_embedding = [tf.nn.embedding_lookup(self.embedding, idx) for idx in sample_idx_each_layer]
        for layer in range(len(self.neighbor_sample_list)):
            input_dim = self.weight_dim_list[layer]
            output_dim =self.weight_dim_list[layer + 1]
            AGG = MeanAggregator(input_dim=input_dim, output_dim=output_dim)
            for v_idx in range(1, len(self.neighbor_sample_list) - layer):
                u_idx = v_idx - 1
                v_sample = self.neighbor_sample_list[u_idx]
                v_sample_embedding = tf.reshape(layer_embedding[v_idx], [-1, v_sample, input_dim]) # [u_nums, num_sample, dim]
                u_sample_embedding = layer_embedding[u_idx] # [u_nums, dim]
                layer_embedding[v_idx - 1] = AGG(u_sample_embedding, v_sample_embedding)

    def sample_for_each_layer(self):
        '''
        :return: [[输入样本点]， [第1层样本点], ... [第len(neighbor_sample_list)层样本点]]
        '''
        u_nodes = self.u_nodes
        sample_idx_each_layer = [u_nodes]
        for i in range(len(self.neighbor_sample_list)):
            sampled_v_nodes = self.sample_from_neighbor(u_nodes, self.neighbor_sample_list[i])
            u_nodes = tf.reshape(sampled_v_nodes, [-1, ])
            sample_idx_each_layer.append(u_nodes)
        return sample_idx_each_layer

    def sample_from_neighbor(self, u_ids, num_samples):
        if num_samples > self.adj_table.shape[-1]:
            raise ValueError('采样数量不能超过邻接表中最大邻居数量')
        with tf.name_scope('sample_from_neighbor'):
            neighbor_idx = tf.nn.embedding_lookup(self.adj_table, u_ids)
            sample_idx = tf.random.shuffle(tf.range(self.adj_table.shape[-1]))[: num_samples]
            sample_neighbor_idx = tf.gather(neighbor_idx, sample_idx, axis=1)
            return sample_neighbor_idx # [batch_size, num_samples]
