#-*- coding: UTF-8 -*-
# @Time    : 2019/3/18 14:45
# @Author  : xiongzongyang
# @Site    : 
# @File    : main.py
# @Software: PyCharm

from process_data import load_data, build_dict, vectorize, load_glove_weights,CNNDataset
from torch.utils.data import DataLoader
from models import attentive_reader
from torch import optim
from torch import nn
import torch
from torch.autograd import Variable


# 获取数据
train_d,train_q,train_a=load_data("/home/xiongzy/GitHome/xiong_local_f_python_uestc2/python3/论文复现/attAttentive Reader/dataset/cnn/train.txt",10000)
dev_d,dev_q,dev_a=load_data("/home/xiongzy/GitHome/xiong_local_f_python_uestc2/python3/论文复现/attAttentive Reader/dataset/cnn/dev.txt")

print("n_train:",len(train_d),"n_dev:",len(dev_d))
print("build dictionary...")
# 构建字典
word_dict=build_dict(train_d+train_q)
entity_markers=list(set([w for w in word_dict.keys() if w.startswith("@entity")]+train_a))
entity_markers=['<unk_entity>']+entity_markers
# 构建实体字典
entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
id2entity={index:w for (index, w) in enumerate(entity_markers)}
print('Entity markers: %d' % len(entity_dict))
num_labels = len(entity_dict)

doc_maxlen = max(map(len, (d for d in train_d)))
query_maxlen = max(map(len, (q for q in train_q)))
print('doc_maxlen:', doc_maxlen, ', q_maxlen:', query_maxlen)

# 序列化
v_train_d, v_train_q,v_train_y = vectorize(train_d, train_q, train_a, word_dict, entity_dict, doc_maxlen,query_maxlen)
v_dev_d, v_dev_q,v_dev_y = vectorize(dev_d, dev_q, dev_a, word_dict, entity_dict, doc_maxlen, query_maxlen)


print('vectroized shape')
print(v_train_d.shape, v_train_q.shape, v_train_y.shape)
print(v_dev_d.shape, v_dev_q.shape, v_dev_y.shape)

vocab_size = max(word_dict.values()) + 1
print('vocab_size:', vocab_size)
embd_size = 100
rnn_half_hidden_size = 64
# 获取与训练词向量

glove_embd_w = load_glove_weights('/home/xiongzy/GitHome/xiong_local_f_python_uestc2/python3/论文复现/attAttentive Reader/dataset/', 100, vocab_size, word_dict)

inputdata=[]
for (d,q,a) in zip(v_train_d,v_train_q,v_train_y):
    inputdata.append((d,q,a))

# 创建模型
# vocab_size,embed_size,rnn_h_size,glove_embd_w,doc_maxlen,query_maxlen,num_labels,pretrain=True,gpu=False
model=attentive_reader(vocab_size,embd_size,rnn_half_hidden_size,glove_embd_w,doc_maxlen,query_maxlen,num_labels)

def train(model):

    # 构建数据集
    traindata=CNNDataset(inputdata)
    dataloader=DataLoader(traindata,batch_size=32,shuffle=True,num_workers=10)

    optimizer=optim.Adam(model.parameters(),lr=0.001)
    loss_function=nn.NLLLoss()
    from tqdm import tqdm
    for epoch in range(100000):
        for ii,(doc,query,ans) in tqdm(enumerate(dataloader)):
            loss=0
            # 准备数据
            doc=Variable(doc)
            query=Variable(query)
            ans=Variable(ans)
            if torch.cuda.is_available():
                doc=doc.cuda()
                query=query.cuda()
                ans=ans.cuda()
                model=model.cuda()
            # 梯度清零
            optimizer.zero_grad()
            # 模型输出
            output=model((doc,query,ans))
            # 计算损失
            loss=loss_function(output,ans.squeeze())
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            if ii%50==0:
                print("epoch:",epoch,"step:",ii,"loss:",loss)

            if ii%100==0:
                predict(model,inputdata[0])
                model.train()


def evaluate(model,input):
    model.eval()
    output=model(input)
    return output

def predict(model,input_line):
    true_label=id2entity[input_line[2][0]]
    doc = input_line[0]
    query = input_line[1]
    ans = input_line[2]
    doc = Variable(torch.Tensor(doc).long().unsqueeze(0))
    query = Variable(torch.Tensor(query).long().unsqueeze(0))
    ans = Variable(torch.Tensor(ans).long())
    if torch.cuda.is_available():
        doc = doc.cuda()
        query = query.cuda()
        ans = ans.cuda()
        model = model.cuda()
    # 传入模型
    output=evaluate(model,(doc,query,ans))

    # topv是top1的值，topi是最大值的索引
    topv,topi=output.data.topk(1)

    # 获取top1的值
    pre_value=topv[0][0]
    label_index=topi[0][0]
    predictions = ""
    for label,ix in entity_dict.items():
        if ix == label_index:
            predictions=label
            break
    print("-"*30)
    print("(%.2f %s %s)"%(pre_value,predictions,true_label))
    print("-" * 30)


if __name__ == '__main__':
    train(model)

