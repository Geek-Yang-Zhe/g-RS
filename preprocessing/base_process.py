import csv
from collections import defaultdict
import time
from itertools import chain

class BaseProcess():

    def generate_sess_date_dict(self, read_path):
        sess_dict = defaultdict(list)  # 每个session {sess_id: [item_id1, item_id2, ...]}
        date_dict = defaultdict(float)  # 每个session结束时间 {sess_id: time}
        with open(read_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                sess_id = row['sessionId']
                sess_dict[sess_id].append((row['itemId'], int(row['timeframe'])))
                date = time.mktime(time.strptime(row['eventdate'], '%Y-%m-%d'))
                if date_dict[sess_id] < date:
                    date_dict[sess_id] = date
        for sess_id in sess_dict.keys():
            sort_sess = sorted(sess_dict[sess_id], key=lambda x: x[1])
            sess_dict[sess_id] = [x[0] for x in sort_sess]
        return sess_dict, date_dict

    # 将item编码
    def encode_sess(self, sess_list, merge_sess=False):
        item2idx = {}
        encoded_sess_list = []
        idx = 1
        for sess in sess_list:
            for item in sess:
                if item not in item2idx.keys():
                    item2idx[item] = idx
                    idx += 1
            encoded_sess_list.append([item2idx[item] for item in sess])
        if merge_sess:
            encoded_item_list = list(chain.from_iterable(encoded_sess_list))
            return encoded_item_list, item2idx
        return encoded_sess_list, item2idx

    # 根据最后一个session的结束时间划分训练集测试集
    def train_test_split(self, sess_dict, date_dict, day=1):
        split_date = max(date_dict.values()) - day * 24 * 3600
        train_sess_list, test_sess_list = [], []
        for sess_id, date in date_dict.items():
            if date > split_date:
                test_sess_list.append(sess_dict[sess_id])
            else:
                train_sess_list.append(sess_dict[sess_id])
        return train_sess_list, test_sess_list