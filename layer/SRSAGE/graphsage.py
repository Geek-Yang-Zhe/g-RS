from layer.GraphSAGE.mean_aggregator import MeanAggregator
import tensorflow as tf

class GraphSAGE():
    def __init__(self, adj_table, nodes_degree, embedding_dim, negative_samples, neighbor_sample_list, weight_dim_list,
                 concat_list, learning_rate=1e-5):
        '''
        :param adj_table: 邻接表，[num_nodes + 1, max_degree]
        :param nodes_degree: 每个节点的实际出度，节点和邻接表中的节点一一对应
        :param embedding_dim:
        :param negative_samples: 负采样个数，用于无监督学习
        :param neighbor_sample_list: GNN中每层被采样的邻居节点数量
        :param weight_dim_list: 每层聚合函数的输出维度
        :param concat_list: 每层聚合函数是否采用concat形式来聚合邻居节点信息和当前节点信息
        '''
        assert len(neighbor_sample_list) == len(weight_dim_list)
        # 不为placeholder，降低IO开销，同时使得其能够参与静态图计算的时候被更改
        self.adj_table = tf.Variable(adj_table, trainable=False, name='adj')
        self.nodes_degree = nodes_degree
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.neighbor_sample_list = neighbor_sample_list
        self.weight_dim_list = [self.embedding_dim] + weight_dim_list
        self.concat_list = concat_list
        self.lr = learning_rate
        self.u_nodes = tf.placeholder(name='u_nodes', shape=[None, ], dtype=tf.int32)
        self.v_nodes = tf.placeholder(name='v_nodes', shape=[None, ], dtype=tf.int32)
        self.agg_list = []
        self.build()

    def build(self):
        with tf.variable_scope("embedding", reuse=True):
            self.embedding = tf.get_variable(name='embedding')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        "链路预测的执行"
        # self._build()
        # self._loss()
        # self._metrics()
        # self._clip_gradients()
        # self.clip_grad_op = self.optimizer.minimize(self.loss)

    def _build(self):
        with tf.name_scope('sample_and_learn_agg_for_u'):
            self.sample_u_idx = self.sample_each_layer(self.u_nodes)
            self.u_embedding, self.agg_list = self.aggregate(self.sample_u_idx) # [batch_size, weight_dim_list[-1]]
        with tf.name_scope('sample_and_agg_for_v'):
            self.sample_v_idx = self.sample_each_layer(self.v_nodes)
            self.v_embedding, _ = self.aggregate(self.sample_v_idx)
        with tf.name_scope('sample_and_agg_for_negatives'):
            self.neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=tf.reshape(tf.cast(self.v_nodes, tf.int64), (-1, 1)),
                num_true=1, num_sampled=self.negative_samples,
                unigrams=self.nodes_degree.tolist(), range_max=len(self.nodes_degree),
                distortion=0.75, unique=False)
            self.sample_neg_idx = self.sample_each_layer(self.neg_samples)
            self.neg_embedding, _ = self.aggregate(self.sample_neg_idx) # [num_neg, weight_dim_list[-1]]
        with tf.name_scope('l2_norm'):
            self.u_embedding = tf.nn.l2_normalize(self.u_embedding, 1)
            self.v_embedding = tf.nn.l2_normalize(self.v_embedding, 1)
            self.neg_embedding = tf.nn.l2_normalize(self.neg_embedding, 1)

    def sage(self):
        with tf.name_scope('sample_and_learn_agg_for_u'):
            self.sample_u_idx = self.sample_each_layer(self.u_nodes)
            self.u_embedding = self.aggregate(self.sample_u_idx) # [batch_size, weight_dim_list[-1]]
        return self.u_embedding

    def _loss(self):
        with tf.name_scope('loss'):
            self.affinity_u_v = tf.reduce_sum(self.u_embedding * self.v_embedding, axis=1)  # [batch_size, ]
            self.affinity_u_neg = tf.matmul(self.u_embedding, self.neg_embedding, transpose_b=True)  # [batch_size, num_neg]
            u_v_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.affinity_u_v), logits=self.affinity_u_v)
            u_neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.affinity_u_neg),
                                                                 logits=self.affinity_u_neg)
            self.loss = tf.reduce_mean(u_v_loss + tf.reduce_sum(u_neg_loss, axis=1))

    def _metrics(self):
        with tf.name_scope('mrr'):
            concat_affinity = tf.concat([tf.reshape(self.affinity_u_v, (-1, 1)), self.affinity_u_neg], axis=1)
            _, top_k_indices = tf.nn.top_k(concat_affinity, k=self.negative_samples + 1)
            self.rank = tf.where(tf.equal(top_k_indices, 0))[:, 1] + 1 # [batch_size, 2]
            self.mrr = tf.reduce_mean(1 / self.rank)

    def _clip_gradients(self):
        grads_and_vars = []
        for gradient, variable in self.optimizer.compute_gradients(self.loss):
            clipped_gradient = None
            if gradient is not None:
                clipped_gradient = tf.clip_by_value(gradient, -5, 5)
            grads_and_vars.append((clipped_gradient, variable))
        self.clip_grad_op = self.optimizer.apply_gradients(grads_and_vars)

    def aggregate(self, sample_idx_each_layer):
        # 第0层聚合：1 -> 0, 2 -> 1, ..., total_layers -> total_layers - 1
        # 第1层聚合：1 -> 0, 2 -> 1, ..., total_layers - 1 -> total_layers - 2
        # 第total_layers - 1层聚合: 1 -> 0
        layer_embedding = [tf.nn.embedding_lookup(self.embedding, idx) for idx in sample_idx_each_layer]
        for layer in range(len(self.neighbor_sample_list)):
            input_dim = self.weight_dim_list[layer]
            output_dim = self.weight_dim_list[layer + 1]
            concat = self.concat_list[layer]
            if layer != 0:
                last_concat = self.concat_list[layer - 1]
                if last_concat:
                    input_dim = input_dim * 2
            if len(self.agg_list) != len(self.neighbor_sample_list):
                if layer != len(self.neighbor_sample_list) - 1:
                    AGG = MeanAggregator(input_dim=input_dim, output_dim=output_dim, layer_idx=layer, concat=concat)
                else:
                    AGG = MeanAggregator(input_dim=input_dim, output_dim=output_dim, layer_idx=layer, concat=concat,
                                         activation=lambda x: x)
                self.agg_list.append(AGG)
            else:
                AGG = self.agg_list[layer]
            for v_idx in range(1, len(sample_idx_each_layer) - layer):
                u_idx = v_idx - 1
                v_sample = self.neighbor_sample_list[u_idx]
                v_sample_embedding = tf.reshape(layer_embedding[v_idx], [-1, v_sample, input_dim])  # [u_nums, num_sample, dim]
                u_sample_embedding = layer_embedding[u_idx]  # [u_nums, dim]
                layer_embedding[v_idx - 1] = AGG((u_sample_embedding, v_sample_embedding))
        return layer_embedding[0]

    def sample_each_layer(self, input_nodes):
        '''
        :return: [[输入样本点]， [第1层样本点], ... [第len(neighbor_sample_list)层样本点]]
        '''
        u_nodes = input_nodes
        sample_idx_each_layer = [u_nodes]
        for i in range(len(self.neighbor_sample_list)):
            # sampled_v_nodes = self.sampler((u_nodes, self.neighbor_sample_list[i]))
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
            return sample_neighbor_idx  # [batch_size, num_samples]
