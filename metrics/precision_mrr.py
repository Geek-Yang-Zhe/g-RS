import tensorflow as tf

class PRE_MRR(object):
    def __init__(self, topk):
        self.topk = topk

    def result(self, y_true, y_pred):
        '''
        :param y_true: 样本真实类别，从0开始编码 [batch_size, ]
        :param y_pred: 每个类别的预测得分, [batch_size, num_class]
        :return: 每个样本的 P@tok_k MRR@top_k 求和
        '''
        _, topk_idx = tf.math.top_k(y_pred, self.topk, sorted=True)
        label = tf.tile(tf.reshape(y_true, [-1, 1]), [1, self.topk]) # [batch_size, topk]
        topk_boolean_mask = tf.equal(topk_idx, label)
        batch_precision = tf.reduce_mean(tf.cast(tf.reduce_any(topk_boolean_mask, axis=-1), tf.float32))
        rank = tf.where(topk_boolean_mask) + 1
        sum_mrr = tf.reduce_sum(1 / rank[:, 1])
        batch_size = tf.reduce_sum(tf.ones_like(y_true, dtype=tf.float64))
        return batch_precision, sum_mrr / batch_size