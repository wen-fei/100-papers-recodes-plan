#-*- coding: UTF-8 -*-
# @Time    : 2019/3/18 14:45
# @Author  : xiongzongyang
# @Site    : 
# @File    : process_data.py
# @Software: PyCharm

from collections import Counter
import numpy as np
from tqdm import tqdm
import torch

import os

# 加载数据
def load_data(in_file,max_example=None,relabeling=True):
    '''
    抽取数据集
    :param in_file:
    :param max_example:
    :param relabeling:
    :return:
    '''
    documents=[]
    questions=[]
    answers=[]
    num_examples=0

    # 读取数据
    print(in_file)
    f=open(in_file,'r',encoding="utf-8")
    while True:
        if num_examples % 10000 ==0:
            print('load_data:n_examples:',num_examples)
        # 读取一行
        line=f.readline()
        if not line: # 如果没有读取到，则推出
            break

        # 依次处理问题，文档，答案
        question=line.strip().lower()
        answer=f.readline().strip()
        document=f.readline().strip().lower()

        if relabeling:
            q_words=question.split(" ")
            d_words=document.split(" ")
            # 判断答案是否是文档中的一个实体
            assert answer in d_words
            entity_dict={}
            entity_id=0
            # 建立原实体到新实体的映射字典
            for word  in d_words+q_words:
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word]='@entity'+str(entity_id)
                    entity_id+=1

            q_words=[entity_dict[w] if w in entity_dict else w for w in q_words]
            d_words=[entity_dict[w] if w in entity_dict else w for w in d_words]
            answer=entity_dict[answer]

        questions.append(q_words)
        answers.append(answer)
        documents.append(d_words)

        num_examples+=1

        f.readline()

        if (max_example is not None) and (num_examples >=max_example):
            break

    f.close()
    return(documents ,questions,answers)

def build_dict(sentences,max_words=50000):
    '''
    建立词汇表
    :param sentences:
    :param max_words:
    :return:
    '''
    word_count=Counter()
    for sent in sentences:
        for w in sent:
            word_count[w]+=1

    # 选择最常见的max_words个词
    ls=word_count.most_common(max_words)
    print("#words:%d->%d"%(len(word_count),len(ls)))
    # 输出词典的样例
    for key in ls[:5]:
        print(key)
    for key in ls[-5:]:
        print(key)

    # w是一个元祖，包含word和词频
    # 0>unk
    # 1>delimiter |||
    return {w[0]:index+1 for (index,w) in enumerate(ls)}

def vectorize(doc,query,ans,word_dict,entity_dict,doc_maxlen,q_maxlen):
    '''
    将文本序列化
    :param doc:
    :param query:
    :param ans:
    :param word_dict:
    :param entity_dict:
    :param doc_maxlen:
    :param q_maxlen:
    :return:
    '''
    in_x1=[]
    in_x2=[]
    in_l=np.zeros((len(doc),len(entity_dict)))
    in_y=[]
    for idx, (d,q,a) in enumerate(zip(doc,query,ans)):
        # 判断答案是否在文档中
        assert (a in d)
        # 将文档中的词转化为序列
        seq1=[word_dict[w] if w in word_dict else 0 for w in d]
        # 过长时，对文档进行截取
        seq1=seq1[:doc_maxlen]
        # 过短时，对文档进行填充
        pad_1=max(0,doc_maxlen-len(seq1))
        seq1+=[0]*pad_1

        # 将问题转化为序列形式
        seq2=[word_dict[w] if w in word_dict else 0 for w in q]
        seq2=seq2[:q_maxlen]
        pad_2=max(0,q_maxlen-len(seq2))
        seq2+=[0]*pad_2

        if (len(seq2)>0) and (len(seq2)>0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            # 将这个文档中出现的实体标记在矩阵中
            in_l[idx,[entity_dict[w] for w in d if w in entity_dict]]=1.0
            # 将标签转化为序列形式
            # y=np.zeros(len(entity_dict))
            if a in entity_dict:
                y=[entity_dict[a]]

            in_y.append(y)
        if idx%10000 ==0:
            print("vectorize:vectorization: processed %d/%d"%(idx,len(doc)))

    return np.array(in_x1),np.array(in_x2),np.array(in_y)

def load_glove_weights(glove_dir,embed_dim,vocab_size,word_index):
    # 加载glove词向量矩阵
    embeddings_index={}
    f=open(os.path.join(glove_dir,'glove.6B.'+str(embed_dim)+"d.txt"))
    for line in tqdm(f,"read glove"):
        values=line.split()
        word=values[0]
        coefs=np.asarray(values[1:],dtype="float32")
        embeddings_index[word]=coefs
    f.close()

    print("found %d word vector"%len(embeddings_index))
    # 创建当前词汇表的词向量矩阵
    embeddding_matrix=np.zeros((vocab_size,embed_dim))
    print("embed_matrix.shape:",embeddding_matrix.shape)
    for word ,i in tqdm(word_index.items(),"build embedding_matrix"):
        # 获取当前词的词向量
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embeddding_matrix[i]=embedding_vector
    # 返回当前词汇表的词向量矩阵
    return embeddding_matrix

import numpy as np
from torch.utils.data import Dataset

class CNNDataset(Dataset):
    def __init__(self,inputdata):
        self.inputdata=inputdata


    def __getitem__(self, idx):
        one_inputdata=self.inputdata[idx]
        doc=one_inputdata[0]
        query=one_inputdata[1]
        ans=one_inputdata[2]
        return np.array(doc),np.array(query),torch.Tensor(ans).long()

    def __len__(self):
        return len(self.inputdata)