import os 
import string
from numpy import loadtxt
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from collections import Counter

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
        #DEBUG CODE:
        if word != 'when':
            idx.append(len(word))
        # else:
        #     print(word)
    if idx == []:
        idx.append(0)
    return idx


def cleanAndTokenizeEDUs(vocab, load_path=None, load_name=None, save_path=None, save_name=None, willReturn = True, max_sent_len = 15,max_doc_len = 6):
    
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
    sent_counter = -1
    
    for i in range(len(lines)):
        
        
        if i == 5:
            a = 1
            pass
        
        
        sent_counter += 1
        
        cleaned = lines[i].replace('\n','').replace('<s>','').replace("'",'')
        cleaned = text_to_word_sequence(cleaned)
        
        if (sent_counter < max_doc_len):

            if (len(cleaned) == 0):
                while (sent_counter < max_doc_len):
                    tokenized = [0]
                    cleaned_doc.append(tokenized)
                    sent_counter += 1
            
            elif (len(cleaned) == 2): # and (i != 0):
                if  (cleaned[0].isnumeric() and cleaned[1].isnumeric()):
                    
                    score = int(cleaned[0])
                    id = int(cleaned[1])
                    scores.append(score)
                    ids.append(id)
                    
                    if (score<=4) or (score>=7):
                        idx_binary.append(doc_count)
                    
                    if (i != 0):
                        filtered_doc = list(filter(None, cleaned_doc))
                        padded_doc = pad_sequences(filtered_doc, maxlen=max_sent_len)
                        
                        cleaned_all.append(padded_doc.tolist())
                        cleaned_doc=[]
                        doc_count += 1
                        sent_counter = 0
                   
                    continue
            
            tokenized=tokenizeListOfTexts(cleaned, vocab)
            cleaned_doc.append(tokenized)
        else: 
            if (len(cleaned) == 2): # and (i != 0):
                if  (cleaned[0].isnumeric() and cleaned[1].isnumeric()):
                    #print(i)
                    
                    score = int(cleaned[0])
                    id = int(cleaned[1])
                    scores.append(score)
                    ids.append(id)
                    
                    if (score<=4) or (score>=7):
                        idx_binary.append(doc_count)
                    
                    if (i != 0):
                        filtered_doc = list(filter(None, cleaned_doc))
                        padded_doc = pad_sequences(filtered_doc, maxlen=max_sent_len)
                        
                        cleaned_all.append(padded_doc.tolist())
                        cleaned_doc=[]
                        doc_count += 1
                        
                    sent_counter = 0
                    continue
            
            
        if (i%100 == 0) and (i!=0):
            print(str(i)+' out of' + str(length))
            #DEBUG CODE
            break
            # print(cleaned_lines[i-500:i])
    
    # Append the last doc 
    
    while (sent_counter < max_doc_len):
        tokenized = [0]
        cleaned_doc.append(tokenized)
        sent_counter += 1
        
    filtered_doc = list(filter(None, cleaned_doc))
    padded_doc = pad_sequences(filtered_doc, maxlen=max_sent_len)
    
    cleaned_all.append(padded_doc.tolist())
        
        # cleaned_doc = cleaned_doc[0]
    # 
    # del cleaned_all[0]
    # del scores[0]
    # del idx_binary[0]
    # del ids[0]
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
        print(save_name + ' cleanAndTokenizeEDUs Saved and Done')

dir = '/home/louis/SharedWindows/edu_data/'  

# def getWordIndices()
# t = Tokenizer()

# tokenized = t.fit_on_texts(cleaned)

        
        ##

dir = '/home/louis/SharedWindows/edu_data/'  
  
# t = Tokenizer()

# tokenized = t.fit_on_texts(cleaned)
if __name__ == '__main__':
    #data=(関数, 引数)
    isReload = int(input("Type 1 to initiate reload of google word2vec: "))
    
    if isReload:
        model_word2vec_temp = gensim.models.KeyedVectors.load_word2vec_format('/home/owner/デスクトップ/milnet+edu/GoogleNews-vectors-negative300.bin', binary=True)  
        model_word2vec = model_word2vec_temp
        vocab = model_word2vec.vocab
    else:
        # For Debug Purposes
        isCounter = int(input("Type 1 to initiate Counter: "))
        if isCounter:
            model_word2vec = Counter()
            vocab = Counter()
            

    print('word2vec model loaded')

    load_path = '/home/louis/SharedWindows/edu_data/'#'/home/owner/デスクトップ/milnet+edu/data/'  
    save_path = '/home/louis/SharedWindows/edu_data/Preprocessed/'#'/home/owner/デスクトップ/milnet+edu/data/Preprocessed/'
      
    load_name = ['imdb-edus.train', 'imdb-edus.test','imdb-edus.dev']
    save_name = ['train_data','test_data','validation_data']
    
    willReturn = False
    
    # cleanAndTokenizeEDUs(model_word2vec.vocab, load_path=load_path, load_name = 'imdb-edus.train', save_path=save_path,save_name=save_name[0], willReturn = False)
    features, scores, idx, ids = cleanAndTokenizeEDUs(vocab, load_path=load_path, load_name = 'imdb-edus.train', save_path=save_path,save_name=save_name[0], willReturn = True)
    
    ##
    
    
    data = [ ( cleanAndTokenizeEDUs, vocab, load_path, load_name[i], save_path, save_name[i] , willReturn) for i in range( len(load_name) ) ]
    with Pool( multi.cpu_count()-1 ) as p:
        p.map( wrapper_func_six, data )
        
    print('All Done')

##
for i in range(len(cleaned_all)):
    print(i)
    for j in range(len(cleaned_all[i])):
        print(j)
