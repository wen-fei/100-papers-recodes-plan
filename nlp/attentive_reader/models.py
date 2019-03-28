#-*- coding: UTF-8 -*-
# @Time    : 2019/3/18 13:37
# @Author  : xiongzongyang
# @Site    : 
# @File    : models.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class attentive_reader(nn.Module):
    def __init__(self,vocab_size,embed_size,rnn_h_size,glove_embd_w,doc_maxlen,query_maxlen,num_labels,pretrain=True,gpu=False):
        super(attentive_reader, self).__init__()
        print("vocab",vocab_size)
        print('embed_size',embed_size)
        print("rnn_h_size",rnn_h_size)
        print('doc_maxlen',doc_maxlen)
        print('query maxlen',query_maxlen)
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.rnn_h_size=rnn_h_size
        self.glove_embed_w=glove_embd_w
        self.doc_maxlen=doc_maxlen
        self.query_maxlen=query_maxlen
        self.query_maxlen=num_labels
        self.pretrain=pretrain
        self.gpu=gpu

        # embedding层,问题和文章是共享的
        self.embedding=nn.Embedding(vocab_size,embed_size)

        # 文章的bilstm层
        self.d_lstm=nn.LSTM(self.embed_size,self.rnn_h_size,batch_first=True,bidirectional=True)

        # 问题的bilstm层
        self.q_lstm=nn.LSTM(self.embed_size,self.rnn_h_size,batch_first=True,bidirectional=True)

        # 定义一个线性层
        self.linear=nn.Linear(rnn_h_size*2,rnn_h_size*2,bias=False)

        # 定义输出层
        self.output=nn.Linear(rnn_h_size*2,num_labels,bias=False)

        # 定义softmax
        self.softmax = nn.LogSoftmax()

        # 初始化嵌入层和矩阵W
        self.init()

    def init(self):
        # 初始化词向量
        # self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.data.copy_(torch.from_numpy(self.glove_embed_w))

        # 初始化线性层
        initrange=0.1
        for p in self.linear.parameters():
            p.data.uniform_(-initrange,initrange)

        for p in self.output.parameters():
            p.data.uniform_(-initrange,initrange)

        # 初始化文章lstm的参数
        for p in self.d_lstm.parameters():
            p.data.uniform_(-initrange, initrange)

        # 初始化问题lstm的参数
        for p in self.q_lstm.parameters():
            p.data.uniform_(-initrange, initrange)

    def forward(self,input):

        # 分离文档，问题和答案
        doc,query,ans=input[0],input[1],input[2]

        # 对文档和问题进行词嵌入
        doc_embed=self.embedding(doc)
        query_embed=self.embedding(query)

        # 文章经过bilstm，这里需要获取他的输出，输出维度为，seq_len,batch,hidden_size*2
        doc_output,doc_hidden=self.d_lstm(doc_embed)

        # 问题经过bilstm,这里需要获取他的隐层状态,
        _,query_hidden=self.q_lstm(query_embed)
        # 分离隐层和细胞状态，维度均为2,batch,hidden_size
        q_h,q_c=query_hidden

        # print(doc_output.shape)
        # print(q_h.shape)

        # 改变q_h的维度
        q_h=q_h.permute(1,0,2)   # 32 2 64
        q_h=q_h.contiguous().view(-1,1,self.rnn_h_size*2)  # batch_size,1,128

        # 计算qw
        qw=self.linear(q_h)
        # print("-"*20)
        # print(qw.shape)

        # 计算qwp
        qwp=qw.mul(doc_output)
        # print(qwp.shape)

        # 计算
        ai=F.softmax(qwp)

        # 计算o
        # print("#"*20)
        o=(ai.mul(doc_output)).sum(1)
        # print(o.shape)

        # 通过线性层
        output=self.output(o)
        output=self.softmax(output)

        return output





