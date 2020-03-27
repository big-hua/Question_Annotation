
import re
import jieba
from utils.config import stopwords_path


def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


def clean_sentence(line):
    line = re.sub(
        "[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '', line)
    words = jieba.cut(line, cut_all=False)
    return words



stop_words = load_stop_words(stopwords_path)


def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    words = clean_sentence(sentence)
    # 过滤停用词
    words = [word for word in words if word not in stop_words]
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def proc(df):
    df['content'] = df['content'].apply(sentence_proc)
    return df