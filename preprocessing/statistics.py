import pandas as pd
import pickle
from itertools import chain

# 统计diginetic中session_id和日期是否是一一对应的，结果不是，有260个session存在跨天
def check_diginetic_unique():
    df = pd.read_csv('../data/train-item-views.csv', sep=';')
    df['eventdate'] = pd.to_datetime(df['eventdate'])
    sess_time_df = df.groupby('sessionId', as_index=False, sort=False)['eventdate'].agg({'date_nunique' : 'nunique'})
    print(sess_time_df[sess_time_df['date_nunique'] != 1])
    # print(sess_time_df['date_nunique'].unique())

def statistics_sess(path='../diginetica'):
    with open(path + '/train.pkl', 'rb') as f:
        train_sess, train_label = pickle.load(f)
    with open(path +'/test.pkl', 'rb') as f:
        test_sess, test_label = pickle.load(f)
    print('训练集session个数为{}，测试集session个数为{}'.format(len(train_sess), len(test_sess)))
    train_len, test_len = 0, 0
    for sess in train_sess:
        train_len += len(sess)
    for sess in test_sess:
        test_len += len(sess)
    print('训练集session平均长度为{}, 测试集session平均长度为{}'.format(train_len / len(train_sess), test_len / len(test_sess)))
    item_set = set(chain.from_iterable(train_sess + test_sess + [train_label] + [test_label]))
    print('item总数为：{}'.format(len(item_set)))

