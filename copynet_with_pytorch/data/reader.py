#-*- coding: UTF-8 -*-
# @Time    : 2019/5/10 14:10
# @Author  : xiongzongyang
# @Site    : 
# @File    : reader.py
# @Software: PyCharm

def read_files(data_path,parse_func=None):
    '''
    input text   output text
    :param data_paths:
    :param parse_func:
    :return:
    '''
    input_seqs=[]
    output_seqs=[]
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            if str(line).strip() is not "":
                input_seq,output_seq=str(line).strip().split('\t')[:2]
                if parse_func!=None:
                    input_seq=parse_func(input_seq)
                    output_seq=parse_func(output_seq)
                input_seqs.append(input_seq)
                output_seqs.append(output_seq)
    return [input_seqs,output_seqs]

if __name__ == '__main__':
    import parser
    input_seqs,output_seqs=read_files('./test.txt',parser.get_word_list)
    print(input_seqs[:3])
    print(output_seqs[:3])