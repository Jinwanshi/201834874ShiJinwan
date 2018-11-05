from textblob import Word
from textblob import TextBlob

import numpy as np
import os
import random
import math
import pickle


#训练集和测试集的划分
def split():
    path = './data/tokenization/'
    for class_ in os.listdir(path):
        files = os.listdir(path+class_+'/')
        random.shuffle(files)
        for file in files[:int(0.8*len(files))]:
            save_file = (path+class_+'/').replace('tokenization', 'train')
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            with open(save_file+file, 'w', encoding='utf-8') as f_w:
                with open(path+class_+'/'+file, 'r', encoding='utf-8') as f:
                    f_w.write(f.readline())

        for file in files[int(0.8*len(files)):]:
            save_file = (path+class_+'/').replace('tokenization', 'test')
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            with open(save_file+file, 'w', encoding='utf-8') as f_w:
                with open(path+class_+'/'+file, 'r', encoding='utf-8') as f:
                    f_w.write(f.readline())


#分词
def tokenizatoin():
    path = './data/dataset/'
    for class_ in os.listdir(path):
        for file in os.listdir(path+class_+'/'):
            file_path = path + class_ + '/' + file
            # print(file_path)
            doc = ''
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    doc += line.strip();
            # print(doc)
            words = TextBlob(doc).words
            words = [word.lower() for word in words]
            temp_words = []
            for word in words:
                flag = True
                for c in word:
                    if not (c == '-' or 'a' <= c <= ''
                                                    ''):
                        flag = False
                        break
                if flag:
                    temp_words.append(word)
            words = temp_words
            words = [Word(word).lemmatize() for word in words]
            words = [Word(word).lemmatize('v') for word in words]

            save_file_path = (path + class_ + '/').replace('dataset', 'tokenization')
            # print(save_file_path)
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)
            with open(save_file_path+file, 'w', encoding='utf-8') as f_w:
                for word in words:
                    f_w.write(word + ' ')

#建立词典
def createVocab():
    path = './data/train/'
    vocab = {}
    for class_ in os.listdir(path):
        # print(class_)
        if class_ == '.DS_Store':
            continue
        for file in os.listdir(path+class_+'/'):
            file_path = path + class_ + '/' + file
            with open(file_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
            for word in line.split(' '):
                vocab[word] = vocab.get(word, 0) + 1
    vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
    stopwords = []
    with open('./data/normalization.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.strip())
    with open('./data/dictionary.txt', 'w', encoding='utf-8') as f_w:
        for word, cnt in vocab:
            if word in stopwords:
                continue
            f_w.write(word + ' ' + str(cnt) + '\n')


def get_information():
    path = './data/information/'
    vocab = []
    with open('./data/dictionary.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip().split(' ')[0])
    # print(len(vocab))
    # 13202
    dic = {}
    for word in vocab:
        dic[word] = {}
        dic[word]['all'] = 0

    path = './data/train/'
    for class_ in os.listdir(path):
        if class_ == '.DS_Store':
            continue
        print(class_)
        for file in os.listdir(path+class_+'/'):
            # print(file)
            with open(path+class_+'/'+file, 'r', encoding='utf-8', errors='ignore') as f:
                words_in_file = f.readline().strip().split(' ')

            for word in dic:
                if word in words_in_file:
                    dic[word]['all'] = dic[word].get('all', 0) + 1

                cnt2 = 0
                for ws in words_in_file:
                    if ws == word:
                        cnt2 += 1
                dic[word][path+class_+'/'+file] = cnt2
    # with open('./data/dic.pkl', 'wb') as f:
    #     pickle.dump(dic, f)
    for word in dic:
        with open('./data/information/'+word+'.pkl', 'wb') as f:
            pickle.dump(dic[word], f)






def vectoring():
    vocab = []
    with open('./data/dictionary.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip().split(' ')[0])

    dic = {}
    for i, word in enumerate(vocab):
        with open('./data/information/' + word + '.pkl', 'rb') as f:
            dic[word] = pickle.load(f)
        print(i)


    count = 0
    path = './data/train/'
    for class_ in os.listdir(path):
        if class_ == '.DS_Store':
            continue
        for file in os.listdir(path+class_+'/'):
            # with open(path+class_+'/'+file, 'r', encoding='utf-8') as f:
            #     words_in_file = f.readline().strip().split(' ')


            vector = []
            for word in vocab:
                idf = dic[word]['all']
                idf = math.log(15056 / idf)
                tf = dic[word][path+class_+'/'+file]
                tf = 1+math.log(tf) if tf != 0 else tf
                # if tf*idf < 0:
                #     print(dic[word][path+class_+'/'+file], tf, idf)
                vector.append(tf*idf)

            save_file = (path + class_).replace('data', 'data/vectoring')
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            with open(save_file+'/'+file+'.txt', 'w', encoding='utf-8') as f_w:
                for v in vector:
                    f_w.write(str(v) + ' ')
            print(count)
            count += 1


def vectoring_test():
    vocab = []
    with open('./data/dictionary.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip().split(' ')[0])

    dic = {}
    for i, word in enumerate(vocab):
        with open('./data/information/' + word + '.pkl', 'rb') as f:
            dic[word] = pickle.load(f)
        print(i)


    count = 0
    path = './data/test/'
    for class_ in os.listdir(path):
        if class_ == '.DS_Store':
            continue
        for file in os.listdir(path+class_+'/'):
            with open(path+class_+'/'+file, 'r', encoding='utf-8') as f:
                words_in_file = f.readline().strip().split(' ')


            vector = []
            for word in vocab:
                idf = dic[word]['all']
                idf = math.log(15056 / idf)

                tf = 0
                for ws in words_in_file:
                    if ws == word:
                        tf += 1

                # tf = dic[word][path+class_+'/'+file]
                tf = 1+math.log(tf) if tf != 0 else tf
                # if tf*idf < 0:
                #     print(dic[word][path+class_+'/'+file], tf, idf)
                vector.append(tf*idf)

            save_file = (path + class_).replace('data', 'data/vectoring')
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            with open(save_file+'/'+file+'.txt', 'w', encoding='utf-8') as f_w:
                for v in vector:
                    f_w.write(str(v) + ' ')
            print(count)
            count += 1


def getTestVector():
    path_test = './data/vectoring/test/'
    for class_test in os.listdir(path_test):
        for file_test in os.listdir(path_test + class_test + '/'):
            with open(path_test + class_test + '/' + file_test, 'r', encoding='utf-8') as f:
                vector = np.asarray(f.readline().strip().split(' '), dtype=np.float)
                yield path_test+class_test+'/'+file_test, np.reshape(vector, [1, -1])


def getTrainVector():
    path_train = './data/vectoring/train/'
    for class_train in os.listdir(path_train):
        for file_train in os.listdir(path_train+class_train+'/'):
            with open(path_train+class_train+'/'+file_train, 'r', encoding='utf-8') as f:
                vector = np.asarray(f.readline().strip().split(' '), dtype=np.float)
                yield path_train+class_train+'/'+file_train, np.reshape(vector, [1, -1])

def cos(v1, v2):
    m = np.sum(np.multiply(v1, v2), axis=1)
    s1 = math.sqrt(np.sum(v1 ** 2, axis=1))
    s2 = math.sqrt(np.sum(v2 ** 2, axis=1))
    return m / (s1 * s2)


def knn():
    paths = test()
    ks = [5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
    for path_test, vector_test in getTestVector():
        if path_test not in paths:
            continue
        dic = {}
        for path_train, vector_train in getTrainVector():
            dis = cos(vector_test, vector_train)
            dic[path_train] = dis
        dic = sorted(dic.items(), key=lambda item: item[1], reverse=True)
        cla = {}
        res = {}
        for i in range(ks[-1]):
            class_ = dic[i][0].split('/')[-2]
            cla[class_] = cla.get(class_, 0) + 1

            if (i+1) in ks:
            # if (i+1) == ks[-1]:
                sort_cla = sorted(cla.items(), key=lambda item: item[1], reverse=True)
                res[i+1] = sort_cla[0][0]
                print('label:', path_test.split('/')[-2], 'predic:', res[i+1])
        with open('./data/result.txt', 'a', encoding='utf-8') as f_w:
            f_w.write(path_test+' '+path_test.split('/')[-2]+' ')
            for key in res:
                f_w.write(str(key)+' '+res[key]+' ')
            f_w.write('\n')


def test():
    path = './data/vectoring/test/'
    paths = []
    for class_ in os.listdir(path):
        if class_ == '.DS_Store':
            continue
        for file in os.listdir(path+class_+'/'):
            paths.append(path+class_+'/'+file)
    random.shuffle(paths)
    return paths[:100]



def a():
    path = './data/result.txt'
    dic = {}
    all_line = 0;
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            all_line += 1
            line = line.strip()
            li  = line.split(' ')
            for i in range(2, len(li), 2):
                if li[i+1] == li[1]:
                    dic[li[i]] = dic.get(li[i], 0) + 1
    dic = sorted(dic.items(), key=lambda item: int(item[0]))
    for d in dic:
        print(d[0], int(d[1]) / all_line)






if __name__ == '__main__':
    # tokenizatoin()
    # split()
    # createVocab()
    # get_information()
    # vectoring()
    # vectoring_test()
    # knn()
    # test()
    a()
    pass