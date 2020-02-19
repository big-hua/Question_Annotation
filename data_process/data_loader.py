import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import root
from utils.multi_proc_utils import parallelize
from utils.params_utils import get_params
from data_process.data_func import proc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer


def data_loader(params, is_rebuild_dataset=False):
    if os.path.exists(os.path.join(root, 'data', 'X_train.npy')) and not is_rebuild_dataset:
        X_train = np.load(os.path.join(root, 'data', 'X_train.npy'))
        X_test = np.load(os.path.join(root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(root, 'data', 'y_test.npy'))
        return X_train, X_test, y_train, y_test

    # 读取数据
    df = pd.read_csv(params.data_path, header=None).rename(columns={0: 'label', 1: 'content'})
    # 并行清理数据
    df = parallelize(df, proc)
    # word2index
    text_preprocesser = Tokenizer(num_words=params.vocab_size, oov_token="<UNK>")
    text_preprocesser.fit_on_texts(df['content'])
    # save vocab
    word_dict = text_preprocesser.word_index
    with open(params.vocab_save_dir + 'voab.txt', 'w', encoding='utf-8') as f:
        for k, v in word_dict.items():
            f.write(f'{k}\t{str(v)}\n')

    x = text_preprocesser.texts_to_sequences(df['content'])
    # padding
    x = pad_sequences(x, maxlen=params.padding_size, padding='post', truncating='post')
    # 划分标签
    df['label'] = df['label'].apply(lambda x: x.split())
    # 多标签编码
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['label'])
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 保存数据
    np.save(os.path.join(root, 'data', 'X_train.npy'), X_train)
    np.save(os.path.join(root, 'data', 'X_test.npy'), X_test)
    np.save(os.path.join(root, 'data', 'y_train.npy'), y_train)
    np.save(os.path.join(root, 'data', 'y_test.npy'), y_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    params = get_params()
    print('Parameters:', params)
    X_train, X_test, y_train, y_test = data_loader(params)
    print(X_train)