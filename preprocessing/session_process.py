import os
import pickle
from preprocessing.base_process import BaseProcess
import csv
from collections import defaultdict
import time

# 将用户点击处理成[session1, session2, ... ], [label1, label2, ... ]
class SessionProcess(BaseProcess):

    def process_diginetica_sample(self, save_path, read_path):
        # 预处理
        sess_dict, date_dict = self.generate_sess_date_dict(read_path=read_path)
        self.filter_sess_dict(sess_dict, date_dict)

        # 划分训练集，测试集，编码
        train_sess_id, test_sess_id = self.train_test_split(date_dict, day=7)
        train_sess_list, item2idx = self.encode_item([sess_dict[sess_id] for sess_id in train_sess_id])
        test_sess_list = self.decode_sess([sess_dict[sess_id] for sess_id in test_sess_id], item2idx)
        print('item的数量为{}'.format(len(item2idx)))

        # 生成序列和预测的标签
        train_sess_list, train_label = self.generate_data_label(train_sess_list)
        test_sess_list, test_label = self.generate_data_label(test_sess_list)

        print(len(train_sess_list))
        print(len(test_sess_list))

        # 写入文件
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + '/train.pkl', 'wb') as f:
            pickle.dump((train_sess_list, train_label), f)
        with open(save_path + '/test.pkl', 'wb') as f:
            pickle.dump((test_sess_list, test_label), f)


    def filter_sess_dict(self, sess_dict, date_dict):
        # 对item_id进行频次过滤，频次过滤之前先删除长度为1的session再统计频次
        for sess_id in list(sess_dict.keys()):
            if len(sess_dict[sess_id]) == 1:
                del sess_dict[sess_id]
                del date_dict[sess_id]
        item_count_dict = {}
        for sess_id in sess_dict.keys():
            for item_id in sess_dict[sess_id]:
                if item_id in item_count_dict:
                    item_count_dict[item_id] += 1
                else:
                    item_count_dict[item_id] = 1
        for sess_id in list(sess_dict.keys()):
            filter_sess = [x for x in sess_dict[sess_id] if item_count_dict[x] >= 5]
            if len(filter_sess) < 2:
                del sess_dict[sess_id]
                del date_dict[sess_id]
            else:
                sess_dict[sess_id] = filter_sess

    def train_test_split(self, date_dict, day=1):
        # 按照session结束时间划分训练集测试集
        sort_date_tuple = sorted(date_dict.items(), key=lambda x: x[1])
        max_date = sort_date_tuple[-1][1] - day * 3600 * 24
        test_sess_id = [sess_id for sess_id, date in sort_date_tuple if date > max_date]
        train_sess_id = [sess_id for sess_id, date in sort_date_tuple if date <= max_date]
        return train_sess_id, test_sess_id

    def decode_sess(self, sess_list, item2idx):
        # 解码，遇到训练集中没有出现的item，忽略，之后判断sequence的长度来决定是否采用这个session。
        decode_sess_list = []
        for sess in sess_list:
            decoder_sess = []
            for item in sess:
                if item in item2idx:
                    decoder_sess.append(item2idx[item])
            if len(decoder_sess) < 2:
                continue
            decode_sess_list.append(decoder_sess)
        return decode_sess_list

    def generate_data_label(self, sess_list):
        # 一个session,[v1,v2,v3]，生成data[[v1,v2],[v1]], label[[v3], [v2]]
        sess_list_exclude_label = []
        label = []
        for sess in sess_list:
            for idx in range(1, len(sess)):
                sess_list_exclude_label.append(sess[:-idx])
                label.append(sess[-idx])
        return sess_list_exclude_label, label


if __name__ == '__main__':
    process = SessionProcess()

