#-*- coding: UTF-8 -*-
# @Time    : 2019/5/10 10:32
# @Author  : xiongzongyang
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

import random
import torch
from torch.utils.data import Dataset
from operator import itemgetter

class Language(object):
    def __init__(self,vocab_limit,data_list,remove_words=[]):
        self.data_list=data_list
        self.remove_words=remove_words
        self.vocab=self.create_vocab()
        print(f"vocab_limit:{vocab_limit}")
        # itemgetter函数用于获取对象的哪些维的数据
        truncated_vocab=sorted(self.vocab.items(),key=itemgetter(1),reverse=True)[:vocab_limit]
        self.tok_to_idx=dict()
        self.tok_to_idx['<MSK>']=0
        self.tok_to_idx['<SOS>']=1
        self.tok_to_idx['<EOS>']=2
        self.tok_to_idx['<UNK>']=3
        for idx,(tok,_) in enumerate(truncated_vocab):
            self.tok_to_idx[tok]=idx+4
        self.idx_to_tok={idx:tok for tok,idx in self.tok_to_idx.items()}

    def create_vocab(self):
        vocab=dict()
        for data in self.data_list:
            for sentence in data:
                for token in sentence:
                    if (not token=="") and (token not in self.remove_words):
                        vocab[token]=vocab.get(token,0)+1
        return vocab

class SequencePairDataset(Dataset):
    def __init__(self,
                 data_list,
                 maxlen=30,
                 lang=None,
                 vocab_limit=None,
                 val_size=0.1,
                 seed=42,
                 is_val=False,
                 use_cuda=False,
                 use_extended_vocab=True,
                 remove_words=[]):
        self.input_seqs=data_list[0]
        self.output_seqs=data_list[1]
        self.maxlen=maxlen
        self.use_cuda=use_cuda
        self.parser=None
        self.val_size=val_size
        self.seed=seed
        self.is_val=is_val
        self.use_extended_vocab=use_extended_vocab

        idxs=list(range(len(self.input_seqs)))
        random.seed(self.seed)
        random.shuffle(idxs)
        num_val=int(len(idxs)*self.val_size)
        if self.is_val:
            self.idxs=idxs[:num_val]
        else:
            self.idxs=idxs[num_val:]
        self.input_seqs=[self.input_seqs[idx] for idx in idxs]
        self.output_seqs=[self.output_seqs[idx] for idx in idxs]

        if lang is None:
            lang=Language(vocab_limit,data_list,remove_words)
        self.lang=lang


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        input_token_list=(['<SOS>']+self.input_seqs[idx]+['<EOS>'])[:self.maxlen]
        output_token_list=(['<SOS>']+self.output_seqs[idx]+['<EOS>'])[:self.maxlen]

        input_seq=self.tokens_to_seq(input_token_list)
        output_seq=self.tokens_to_seq(output_token_list,input_token_list=input_token_list)

        if self.use_cuda:
            input_seq=input_seq.cuda()
            output_seq=output_seq.cuda()

        return input_seq,output_seq,"".join(input_token_list),"".join(output_token_list)
    def tokens_to_seq(self,token_list,input_token_list=None):
        seq=torch.zeros(self.maxlen).long()
        tok_to_idx_extension=dict()
        for pos,token in enumerate(token_list):
            if token in self.lang.tok_to_idx:
                # token in vocab
                idx=self.lang.tok_to_idx[token]
            elif token in tok_to_idx_extension:
                # token in vocab_extension
                idx=tok_to_idx_extension[token]
            elif self.use_extended_vocab and input_token_list is not None:
                tok_to_idx_extension[token]=tok_to_idx_extension.get(token,next((pos+len(self.lang.tok_to_idx) for pos, input_token in enumerate(input_token_list) if input_token==token),3))
                idx=tok_to_idx_extension[token]
            elif self.use_extended_vocab:
                idx=pos+len(self.lang.tok_to_idx)
            else:
                idx=self.lang.tok_to_idx['<UNK>']
            seq[pos]=idx
        return seq

    def seq_to_string(self,seq,input_tokens=None):
        '''
        :param seq:
        :param input_tokens:
        :return:
        '''
        vocab_size=len(self.lang.idx_to_tok)
        seq_length=(seq!=0).sum()
        words=[]
        for idx in seq[:seq_length]:
            idx=int(idx.cpu().numpy())
            if idx<vocab_size:
                words.append(self.lang.idx_to_tok[idx])
            elif input_tokens is not None:
                words.append(input_tokens[idx-vocab_size])
            else:
                words.append("<???>")
        string="".join(words)
        return string

if __name__ == '__main__':
    import parser
    import reader
    data_list=reader.read_files("./data_1000.txt",parser.get_word_list)
    spdataset=SequencePairDataset(data_list,vocab_limit=100)
    x=spdataset.__getitem__(1)
    print(x)
