#-*- coding: UTF-8 -*-
# @Time    : 2019/5/10 9:35
# @Author  : xiongzongyang
# @Site    : 
# @File    : create_data.py
# @Software: PyCharm


data_path="./eng-fra.txt"
result=[]
# 打开文件，并读取每一行
with open(data_path,"r",encoding="utf-8") as f:
    file_list=f.readlines()
for line in file_list:
    en_sentence=str(line).strip().split("\t")[0]
    result.append(en_sentence)

# 写入文件
file_len=len(result[:1000])
with open(f"./data_{file_len}.txt", "w", encoding="utf-8") as fw:
    for i in range(0,file_len):
        fw.write(str(result[i]).strip()+"\t"+str(result[i]).strip()+"\n")
