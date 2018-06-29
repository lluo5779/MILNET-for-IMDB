import os 
import string
from numpy import loadtxt
from keras.preprocessing.text import Tokenizer,text_to_word_sequence

import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.pardir)
import glob
import numpy as np
from multiprocessing import Pool
import multiprocessing as multi
sys.path.append(os.pardir)
import pandas as pd
import h5py
import re
import gensim
# lines = loadtxt("/home/louis/SharedWindows/edu_data/"+"imdb-edus.train")
# train = []

# with open(dir+"imdb-edus.train", 'r') as train_file:
#     lines = train_file.readlines()


##



# '''text2index'''
# 
# from func import TextDeal
# from func import wrapper_func_four
# 
# vocab = wList
# 
# td = TextDeal()
# load_path = sorted( glob.glob( '/home/dl-box/デスクトップ/PythonFile/hirose/Text/text_list/*.txt' ) )
# name = [ os.path.splitext( os.path.basename( r ) )[0] for r in load_path ]
# save_path = '/home/dl-box/デスクトップ/PythonFile/hirose/LSTM/W2I/'


##

def wrapper_func_six(tuple_data):
    return(tuple_data[0](tuple_data[1],tuple_data[2],tuple_data[3],tuple_data[4], tuple_data[5], tuple_data[6]) )

def readEduStyledFiles(dir, filename):
    with open(dir+filename, 'r') as train_file:
        lines = train_file.readlines()
    return lines
        
def tokenizeListOfTexts(list, vocab):
    idx = []
    for word in list:
        if word in vocab:
            idx.append(vocab[word].index)
            #print(word)
        # else:
        #     print(word)
    return idx


def cleanAndTokenizeEDUs(vocab, load_path=None, load_name=None, save_path=None, save_name=None, willReturn = True):
    
    if (willReturn == False) and (all(v is None for v in [load_path, load_name, save_path, save_name])):
        raise ValueError('Function Arguments cannot all be None Type when saving files')
    elif (willReturn == True) and (all(v is None for v in [load_path, load_name])):
        raise ValueError('Load arguments cannot all be None Type when loading file')
    
    lines = readEduStyledFiles(load_path,load_name)

    cleaned_lines = []
    length = len(lines)
    scores = []
    ids = []
    cleaned_doc=[]
    cleaned_all=[]
    idx_binary = [] # index for binary training only
    doc_count = 0
    
    for i in range(len(lines)):
        cleaned = lines[i].replace('\n','').replace('<s>','').replace("'",'')
        cleaned = text_to_word_sequence(cleaned)
        
        
        if (len(cleaned) == 2): # and (i != 0):
            if not (cleaned[0].isalpha() or cleaned[1].isalpha()):
                score = int(cleaned[0])
                id = int(cleaned[1])
                if (score<=4) or (score>=7):
                    idx_binary.append(doc_count)
                filtered_doc = list(filter(None, cleaned_doc))
                scores.append(score)
                ids.append(id)
                cleaned_all.append(filtered_doc)
                cleaned_doc=[]
                doc_count += 1
                continue
        
        tokenized=tokenizeListOfTexts(cleaned, vocab)
        
        if (i%10000 == 0) and (i!=0):
            print(str(i)+' out of' + str(length))
            break
            # print(cleaned_lines[i-500:i])
    
        cleaned_doc.append(tokenized)
        # cleaned_doc = cleaned_doc[0]
    
    features = np.asarray(cleaned_all)
    labels = np.asarray(scores)
    idx = np.asarray(idx_binary)
    
    if (willReturn == True):
        print('cleanAndTokenizeEDUs Returned and Done')
        return features, labels, idx, ids
    else:
        np.save(save_path + save_name + '_features'+ '.npy', features)
        np.save(save_path + save_name + '_labels' + '.npy', labels)
        np.save(save_path + save_name + '_idx_binary_only' + '.npy', idx)
        np.save(save_path + save_name + '_doc_ids' + '.npy', ids)
        print('cleanAndTokenizeEDUs Saved and Done')

dir = '/home/louis/SharedWindows/edu_data/'  
  
# t = Tokenizer()

# tokenized = t.fit_on_texts(cleaned)


if __name__ == '__main__':
    #data=(関数, 引数)
    isReload = int(input("Type 1 to initiate reload of google word2vec: "))
    
    if isReload:
        model_word2vec_temp = gensim.models.KeyedVectors.load_word2vec_format('/home/louis/SharedWindows/GoogleNews-vectors-negative300.bin', binary=True)  
        model_word2vec = model_word2vec_temp

    print('word2vec model loaded')

    load_path = '/home/louis/SharedWindows/edu_data/'  
    save_path = '/home/louis/SharedWindows/edu_data/Preprocessed/'
      
    load_name = ['imdb-edus.train', 'imdb-edus.test']
    save_name = ['train','test']
    
    willReturn = False
    
    # cleanAndTokenizeEDUs(model_word2vec.vocab, load_path=load_path, load_name = 'imdb-edus.train', save_path=save_path,save_name=save_name[0], willReturn = False)
    # features, scores, idx, ids = cleanAndTokenizeEDUs(model_word2vec.vocab, load_path=load_path, load_name = 'imdb-edus.train', save_path=save_path,save_name=save_name[0], willReturn = False)
    
    vocab = model_word2vec.vocab
    data = [ ( cleanAndTokenizeEDUs, vocab, load_path, load_name[i], save_path, save_name[i] , willReturn) for i in range( len(load_name) ) ]
    with Pool( multi.cpu_count()-1 ) as p:
        p.map( wrapper_func_six, data )

