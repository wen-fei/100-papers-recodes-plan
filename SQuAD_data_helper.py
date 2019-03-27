#-*- coding: UTF-8 -*-
# @Time    : 2019/3/27 20:55
# @Author  : xiongzongyang
# @Site    : 
# @File    : SQuAD_data_helper.py
# @Software: PyCharm

import json
import nltk

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

def preprocess_file(path):
    # 创建一个结果list
    dump = []
    # 定义一些非法字符
    abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 获取文件数据，得到的是一个list，list里面的每个元素是一个字典，一个字典包含文章问题和答案信息
        data = data['data']

        # 对于文件数据中的每一篇文章
        for article in data:
            # 获取段落字段，对于每一个段落，首先获取段落内容，然后获取段落的问答对
            for paragraph in article['paragraphs']:
                ## 获取段落内容
                context = paragraph['context']
                ## 将段落内容分词，即获取文本的所有单词
                tokens = word_tokenize(context)
                # 获取问答对
                for qa in paragraph['qas']:
                    # 获取问题的id号
                    id = qa['id']
                    # 获取问题
                    question = qa['question']
                    # 获取答案list,遍历这个list，对于一个答案，就可以构造出一条数据
                    for ans in qa['answers']:
                        # 处理答案内容，起始位置，结束位置
                        answer = ans['text']
                        s_idx = ans['answer_start']
                        e_idx = s_idx + len(answer)

                        l = 0
                        s_found = False
                        # 对于每一个单词
                        for i, t in enumerate(tokens):
                            # 遇到第一个正常字符退出循环
                            while l < len(context):
                                if context[l] in abnormals:
                                    l += 1
                                else:
                                    break
                            # exceptional cases
                            if t[0] == '"' and context[l:l + 2] == '\'\'':
                                t = '\'\'' + t[1:]
                            elif t == '"' and context[l:l + 2] == '\'\'':
                                t = '\'\''

                            l += len(t)
                            if l > s_idx and s_found == False:
                                s_idx = i
                                s_found = True
                            if l >= e_idx:
                                e_idx = i
                                break

                        # 将这个答案所对应的数据加入结果集
                        dump.append(dict([('id', id),
                                          ('context', context),
                                          ('question', question),
                                          ('answer', answer),
                                          ('s_idx', s_idx),
                                          ('e_idx', e_idx)]))

    with open(f'{path}_preprocessed', 'w', encoding='utf-8') as f:
        for line in dump:
            json.dump(line, f)
            print('', file=f)

if __name__ == '__main__':
    data_path=""
    preprocess_file(data_path)