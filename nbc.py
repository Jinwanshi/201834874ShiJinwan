import json
import time
import nltk
import os
import numpy as np
import chardet
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from collections import Counter
import sklearn

def if_reserve(raw_word):
    """
    judging whether the tokenized words are reserve
    reserve --> true
    drop --> false
    """
    if bool(re.search(r'\d', raw_word)):
        return  False
    else:
        # words after tokenization are still mixed with invalid term
        # such as 14//90, _great, v.32bis/v.42bis
        # this step is to clear digits or signs in word
        word=re.sub("[^A-Za-z]","",raw_word)
        # clear the blank
        if len(word) <= 1:
            return False
        else:
            return True

def word_process(file_name):
    '''
    :param file_name: text file name
    text different encoding type makes open file failing
    detect the encoding type --> decoding
    :return: bag of word
    '''
    # decode file
    reader = open(file_name, 'rb').read()
    encoding_type = chardet.detect(reader)['encoding']
    if encoding_type == None:
        encoding_type = 'utf8'
    text = reader.decode(encoding=encoding_type)
    text = text.lower()

    ## tokenization : split sentence into words & other signs
    tokenize = nltk.word_tokenize(text)

    tokenize_enhance=[re.sub("[^A-Za-z]","",x) for x in tokenize if if_reserve(x)]

    lemmatizer = WordNetLemmatizer()
    lemmatization = [lemmatizer.lemmatize(x) for x in tokenize_enhance]

    stopword = [x for x in lemmatization if x not in stopwords.words('english')]
    unknowword = [x for x in stopword if x in word_dict]
    # print('bags of word:', unknowword)
    return unknowword

with open('./test-3759.json') as f:
    test = json.load(f)
with open('./train-15069.json') as f:
    train = json.load(f)

def data_process():
    t1=time.clock()
    data_dict={}
    print('Processing training data. Waiting about 700s')
    for i in train:
        class_i=i.split('/')[0]
        if class_i not in data_dict.keys():
            data_dict[class_i]=[i]
        else:
            data_dict[class_i]=data_dict[class_i]+[i]
    # print(data_dict.keys())
    class_dict={}
    for i in data_dict:
        file_word=[]
        for file in data_dict[i]:
            words=word_process('H:/heyFighting/GitHub/201814809dongxue/data/'+file)
            file_word.append(words)
        class_dict[i]=file_word
    print('Processing training data spends: %ds'%int(time.clock()-t1))
    return class_dict

def probability_calculation(class_dict):
    P_aiyj={}
    P_yi={}
    for class_i in class_dict:
        # print(class_i)
        P_yi[class_i]=len(class_dict[class_i])/len(train)
        class_i_word=[]
        for i in class_dict[class_i]:
            class_i_word+=i
        for word in word_dict:
            P_aiyj[word+'-'+class_i]=(class_i_word.count(word)+1)/(len(class_i_word)+1)

    return P_yi, P_aiyj

def bayesian_classify(P_yi, P_aiyj):
    print('Processing testing data. Waiting about 200s')
    t1=time.clock()
    test_label = []
    test_matrix = []
    for i in test:
        test_label.append(i.split('/')[0])
        test_matrix.append(set(word_process('H:/heyFighting/GitHub/201814809dongxue/data/'+i)))
    print('Processing testing data spends %ds'%int(time.clock()-t1))
    # print(test_matrix)
    print('Start testing ')
    accuracy=0
    predict_label=None
    for test_idx in range(len(test_matrix)):
        max_probability = -10000
        test_i=test_matrix[test_idx]
        for class_i in set(test_label):
            probability_i=1
            for word_i in test_i:
                # print('P_aiyj',P_aiyj[word_i+'-'+class_i])
                probability_i += np.log(P_aiyj[word_i+'-'+class_i])

            probability_i+=np.log(P_yi[class_i])
            # print('P_yi', P_yi[class_i])
            if probability_i > max_probability:
                max_probability=probability_i
                predict_label=class_i

        # print(max_probability)

        if predict_label==test_label[test_idx]:
            accuracy+=1
    print('Test accuracy:',accuracy/len(test_label))


# dict_file=['./dict-50-6010.json','./dict-45-6442.json','./dict-40-6991.json',
#            './dict-35-7660.json','./dict-30-8533.json','./dict-25-9677.json',
#            './dict-20-11160.json','./dict-15-13537.json','./dict-10-17396.json']
dict_file = ['./dict-3-35611.json','./dict-7-21618.json']
for i in dict_file:
    with open(i, 'r') as f:
        word_dict = json.load(f)
    print('--------------------Dictionary:',i,'--------------------')

    class_dict=data_process()
    P_yi, P_aiyj=probability_calculation(class_dict)

    bayesian_classify(P_yi, P_aiyj)