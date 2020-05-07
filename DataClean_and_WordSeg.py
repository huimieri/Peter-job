import pandas as pd
import os
import collections
import random
import numpy as np
import jieba
from jieba import posseg
import re


REMOVE_WORDS=['|','[',']','语音','图片']


#读取停用字符
def read_stopwords(path):
    lines=set()
    with open(path,'r',encoding='utf-8')as f:
        for line in f:
            line=line.strip()
            lines.add(line)
    return lines
#出去我们上面自己定义的停用字符
def remove_words(words_list):
    words_list=[word for word in words_list if word not in REMOVE_WORDS]


#分词
def segment(sentence,cut_type='word',pos=False):
    if pos:
        if cut_type=='word':
            word_pos_seq=posseg.lcut(sentence)
            word_seq,pos_seq=[],[]
            for w,p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq,pos_seq
        elif cut_type=='char':
            word_seq=list(sentence)
            pos_seq=[]
            for w in word_seq:
                w_p=posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq,pos_seq
    else:
        if cut_type=='word':
            return jieba.lcut(sentence)
        elif cut_type=='char':
            return list(sentence)


#数据处理
def parse_data(train_path,test_path):
    train_data=pd.read_csv(train_path,encoding='utf-8')
    #去除Report为空的样本
    train_data.dropna(subset=['Report'],how='any',inplace=True)
    #填充x为空的样本
    train_data.fillna('',inplace=True)
    #把表格中的Question和Dialogue拼接起来作为x
    train_x=train_data.Question.str.cat(train_data.Dialogue)
    train_y=train_data.Report
    assert len(train_x) == len(train_y)
    test_data=pd.read_csv(test_path,encoding='utf-8')
    test_data.fillna('',inplace=True)
    test_x=test_data.Question.str.cat(test_data.Dialogue)
    test_y=[]
    return train_x,train_y,test_x,test_y


def save_data(data_1,data_2,data_3,data_path_1,data_path_2,data_path_3,stopwords_path):
    stopwords=read_stopwords(stopwords_path)
    
    with open(data_path_1,'w',encoding='utf-8') as f1:
        count_1=0
        for line in data_1:
            if isinstance(line,str):
                seg_list=segment(line.strip(),cut_type='word')
                seg_list=[word for word in seg_list if word not in stopwords]
                if len(seg_list)>0:
                    seg_line=' '.join(seg_list)
                    f1.write('%s' % seg_line)
                    f1.write('\n')
                    count_1+=1
        print('trainx_length=',count_1)
        
    with open(data_path_2,'w',encoding='utf-8') as f2:
        count_2=0
        for line in data_2:
            if isinstance(line,str):
                seg_list=segment(line.strip(),cut_type='word')
                seg_list=[word for word in seg_list if word not in stopwords]
                if len(seg_list)>0:
                    seg_line=' '.join(seg_list)
                    f2.write('%s' % seg_line)
                    f2.write('\n')
                    count_2+=1
                else:
                    f2.write('随时 联系'+'\n')
        print('trainy_length=',count_2)
    
    with open(data_path_3,'w',encoding='utf-8') as f3:
        count_3=0
        for line in data_3:
            if isinstance(line,str):
                seg_list=segment(line.strip(),cut_type='word')
                seg_list=[word for word in seg_list if word not in stopwords]
                if len(seg_list)>0:
                    seg_line=' '.join(seg_list)
                    f3.write('%s' % seg_line)
                    f3.write('\n')
                    count_3+=1
        print('train_length=',count_3)


if __name__ == '__main__':
    train_list_src, train_list_trg, test_list_src, _ = parse_data('data/data31501/TrainSet.csv','data/data31501/TestSet.csv')
    print(len(train_list_src))
    print(len(train_list_trg))
    save_data(train_list_src,train_list_trg,test_list_src,'train_set.seg_x.txt','train_set.seg_y.txt','test_set.seg_x.txt',stopwords_path='stop_words.txt')







