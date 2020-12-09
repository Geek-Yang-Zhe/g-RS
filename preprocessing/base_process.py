import csv
from collections import defaultdict
import time

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

    def encode_item(self, sess_list):
        item2idx = {}
        encoded_sess_list = []
        idx = 1
        for sess in sess_list:
            for item in sess:
                if item not in item2idx.keys():
                    item2idx[item] = idx
                    idx += 1
            encoded_sess_list.append([item2idx[item] for item in sess])
        return encoded_sess_list, item2idx