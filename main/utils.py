import numpy as np
from itertools import chain

def padding_mask(data, label):
    maxlen = max([len(sess) for sess in data])
    mask = [[1] * len(sess) + [0] * (maxlen - len(sess)) for sess in data]
    padding_sess = [sess + [0] * (maxlen - len(sess)) for sess in data]
    return np.array(mask), np.array(padding_sess), np.array(label)

def get_unique_item_nums(data):
    if isinstance(data, np.ndarray):
        return np.unique(data).shape[0]
    elif isinstance(data, list):
        flatten_data = chain.from_iterable(data)
        return len(set(flatten_data))
    else:
        raise TypeError('传参类型错误，必须为np.array或者list(list)类型参数')

def generate_batch_index(total_samples, batch_size, seed):
    np.random.seed(seed)
    index = np.random.permutation(np.arange(total_samples))
    num_batches = (total_samples - 1) // batch_size + 1
    batch_index_list = []
    for i in range(num_batches):
        batch_index_list.append(index[i * batch_size: min((i + 1) * batch_size, total_samples)])
    return batch_index_list

def encoder(input):
    vocab = {}
    for item in input:
        if item not in vocab:
            vocab[item] = len(vocab)
    return vocab

def generate_batch_AdjMatrix(input):
    if not isinstance(input, np.ndarray) or input.ndim != 2:
        raise TypeError('输入应该为padding后的二维数组')
    batch_in_adj_matrix, batch_out_adj_matrix, batch_unique_item, batch_input_index = [], [], [], []
    "会话内unique最大长度，包括0"
    max_node = max([len(np.unique(sess)) for sess in input])
    for sess in input:
        "对每个序列进行编码，本质上为了压缩图"
        item_vocab = encoder(sess)
        padding_item = list(item_vocab.keys()) + [0] * (max_node - len(item_vocab))
        batch_unique_item.append(padding_item)
        adj_matrix = np.zeros((max_node, max_node))
        for idx in range(len(sess) - 1):
            if sess[idx + 1] == 0:
                break
            u = item_vocab[sess[idx]]
            v = item_vocab[sess[idx + 1]]
            adj_matrix[u, v] = 1
        out_matrix = adj_matrix
        in_matrix = adj_matrix.transpose()
        out_degree = np.clip(np.sum(out_matrix, axis=1, keepdims=True), 1, None)
        in_degree = np.clip(np.sum(in_matrix, axis=1, keepdims=True), 1, None)
        batch_out_adj_matrix.append(np.divide(in_matrix, in_degree))
        batch_in_adj_matrix.append(np.divide(out_matrix, out_degree))
        "原始会话中，每个节点对应的编码"
        batch_input_index.append([item_vocab[item] for item in sess])
    return np.array(batch_out_adj_matrix), np.array(batch_in_adj_matrix), np.array(batch_unique_item), np.array(
        batch_input_index)
