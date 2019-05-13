#-*- coding: UTF-8 -*-
# @Time    : 2019/5/10 13:59
# @Author  : xiongzongyang
# @Site    : 
# @File    : parser.py
# @Software: PyCharm

import jieba

# 分词
def get_word_list(sentence):
    wordlist=list(jieba.cut(sentence))
    return wordlist

# 获取单字list
def get_char_list(sentence):
    charlist=list(sentence)
    return charlist

if __name__ == '__main__':
    sentence="今天的天气很好"
    print(get_word_list(sentence))
    print(get_char_list(sentence))