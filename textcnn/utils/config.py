import os
import pathlib

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

stopwords_path = os.path.join(root, 'data', 'stopwords/哈工大停用词表.txt')

data_path = os.path.join(root, 'data', 'baidu_95.csv')
vocab_save_dir = os.path.join(root, 'data/')
result_path = os.path.join(root, 'results/')