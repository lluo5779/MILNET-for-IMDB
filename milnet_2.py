import sys, os
sys.path.append(os.pardir)

os.environ["KERAS_BACKEND"]='tensorflow'
#import glob
import numpy as np
#from multiprocessing import Pool
#import multiprocessing as multi
#from data.func import load_npy, padding_mat
#sys.path.append('C:\\ProgramData\\Anaconda3\\pkgs\\pydot-1.2.3-py36hd4f83f9_0\\Lib\\site-packages')
#sys.path
x_train = np.load('/home/louis/SharedWindows/x_train_sort.npy')
x_test = np.load('/home/louis/SharedWindows/x_test_sort.npy')
y_train = np.load('/home/louis/SharedWindows/t_train.npy')
y_test = np.load('/home/louis/SharedWindows/t_test.npy')

embWeights=np.load('/home/louis/SharedWindows/weights.npy')
idx=np.load('/home/louis/SharedWindows/index.npy')
embWeights = embWeights[idx]

print('data loaded')
import keras
from keras.layers import Input, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import keras.backend as K
from keras.layers import Lambda, regularizers, Average

from keras.models import Model
from keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, GlobalMaxPooling2D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.layers import merge, Layer, Activation, Dot, Concatenate, Flatten, Lambda



numSentencesPerDoc, numWordsPerSentence = x_train[0].shape[0], x_train[0].shape[1]
print(numSentencesPerDoc, numWordsPerSentence)
#print(x_train[0])

vocabSize, embeddingSize = embWeights.shape[0], embWeights.shape[1]
print(vocabSize, embeddingSize)

dropWordEmb = 0.25
recursiveClass = GRU

filters = 1 #embeddingSize*2
windowMin = 2
windowMax = 4
dimOfSentimentMetrics = 5
batch_size = 1250

##

#wordsInputs = Input(shape=(numWordsPerSentence,1), batch_shape=(numSentencesPerDoc,numWordsPerSentence,), dtype='int32', name='words_input')

x_in = Input( shape = ( numSentencesPerDoc, numWordsPerSentence ) , name='Input' )
#x_pop = Lambda( lambda x: x, output_shape=(numWordsPerSentence, ) , name='convert_shape' )( x_in )
    
#Layer functionの定義
embLayer = Embedding( input_dim=embWeights.shape[0], output_dim=embWeights.shape[1], weights=[embWeights]
                      ,mask_zero=True , trainable=False, embeddings_regularizer=regularizers.l2(0.0000001)
                      , input_length=numWordsPerSentence, name='Embedding' )
#emb = Embedding(vocabSize, embeddingSize, mask_zero=True, weights=[embWeights], 
#              trainable=False, name='embedding')(wordsInputs)
# 
# if (dropWordEmb != 0.0):
#     emb = Dropout(dropWordEmb, name='embed_dropout')(emb)
# 
# newShape= (K.shape(emb)[0],numWordsPerSentence,embeddingSize,numSentencesPerDoc,) #(?, 10, 50, 300)
# reshaped = Lambda(lambda x: K.reshape(x,shape=newShape), name ='Extra_dim_for_convo')(emb)

maxPooledPerDoc = []
convNets = []

filters = 1

for windowSize in range(windowMin,windowMax):
    name='word_mat_convo'+str(windowSize)
    convNet = Conv2D(filters, kernel_size=(windowSize,embeddingSize), padding='valid', 
                           activation='relu', strides=1, use_bias=True, 
                           name=name)
    convNets.append(convNet)
    
for i in range(numSentencesPerDoc):
    maxPooledPerSentence = []
    for j in range(windowMax-windowMin):   
        x_pop = Lambda(lambda x: x[:,i], output_shape=(numWordsPerSentence, ) , name='convert_shape'+str(i+1) )( x_in )
        emb = embLayer(x_pop)
        newShape = (-1,int(emb.shape[1]),int(emb.shape[2]),1)
        reshaped = Lambda(lambda x: K.reshape(x,shape=newShape), name ='Extra_dim_for_convo')(emb)
        wordsCNN  = convNets[j](reshaped)
        #print(wordsCNN.shape)
        # wordsCNN = Flatten()(wordsCNN)
    
        squeezed = Lambda(lambda x: K.squeeze(x, 3))(wordsCNN)#K.squeeze(wordsCNN, 3)
        #print(squeezed)
    
        wordsCNNPooled= MaxPooling1D(pool_size = int(squeezed.shape[1]), padding='valid')(squeezed)
        #print(wordsCNNPooled.shape)
        #squeezed2 = Lambda(lambda x: K.squeeze(x, 2))(wordsCNNPooled)
        #print(squeezed2.shape)
    
    
        maxPooledPerSentence.append(wordsCNNPooled)
        
    mergedPoolForSentence = Concatenate(axis = 1)(maxPooledPerSentence)
    newShape=(-1,1,int(mergedPoolForSentence.shape[1]))
    mergedPoolForSentence = Lambda(lambda x: K.reshape(x,shape=newShape), name ='switch_axis_for_Dense')(mergedPoolForSentence)
    mergedPoolForSentence = Dropout(0.5)(Dense(6, activation='softmax', use_bias=True)(mergedPoolForSentence))

    maxPooledPerDoc.append(mergedPoolForSentence)
    
#Apply Attention 
mergedPoolPerDoc = mergedPoolForSentence = Concatenate(axis = 1)(maxPooledPerDoc)
biRnn = GRU(6,  return_sequences=True)(mergedPoolPerDoc)
newShape = (-1, int(mergedPoolPerDoc.shape[1]), int(mergedPoolPerDoc.shape[2]))
biRnn = Lambda(lambda x: K.reshape(x,shape=newShape), name ='biRnn_TF_Reminder')(emb)

CONTEXT_DIM = int(int(biRnn.shape[1])*int(biRnn.shape[2])/10) 

eij = Dense(CONTEXT_DIM, use_bias=True, activation='tanh')(biRnn)
eij = Dense(CONTEXT_DIM, use_bias=False, activation='softmax')(eij)

weighted_input_ = merge([eij, biRnn], mode='dot', dot_axes=1)
weighted_input = Lambda(lambda x: K.reshape(x,shape=(-1,int(weighted_input_.shape[1])*int(weighted_input_.shape[2]))), name ='attend_output')(weighted_input_)

out = Dense(1, activation='softmax', use_bias=True)(weighted_input)

# NOT NEEDED
# CONTEXT_DIM = int(int(biRnn.shape[1])*int(biRnn.shape[2])/2)
# 
# class AttLayer(Layer):
#     def __init__(self, regularizer=None, **kwargs):
#         self.regularizer = regularizer
#         self.supports_masking = True
#         super(AttLayer, self).__init__(**kwargs)
# 
#     def build(self, input_shape):
#         assert len(input_shape) == 3        
#         self.W = self.add_weight(name='W', shape=(input_shape[-1], CONTEXT_DIM), initializer='normal', trainable=True, 
#                                  regularizer=self.regularizer)
#         self.b = self.add_weight(name='b', shape=(CONTEXT_DIM,), initializer='normal', trainable=True, 
#                                  regularizer=self.regularizer)
#         self.u = self.add_weight(name='u', shape=(CONTEXT_DIM,), initializer='normal', trainable=True, 
#                                  regularizer=self.regularizer)        
#         super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
# 
#     def call(self, x, mask=None):
#         eij = K.dot(K.tanh(K.dot(x, self.W) + self.b), K.expand_dims(self.u))
#         ai = K.exp(eij)
#         alphas = ai / K.sum(ai, axis=1)
#         if mask is not None:
#             # use only the inputs specified by the mask
#             alphas *= K.expand_dims(mask)
#         weighted_input = K.dot(K.transpose(x), alphas)
#         return K.reshape(weighted_input, (weighted_input.shape[0],))
# 
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[-1])
#     
#     def get_config(self):
#         config = {}
#         base_config = super(AttLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
# 
#     def compute_mask(self, inputs, mask):
#         return None

##
model = Model(input=[x_in], output=[out])
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("Model Build Complete")

##
print('Train...')
history = model.fit(x_train, y_train, batch_size=256, verbose=1, epochs=100
                    ,validation_split=0.2, shuffle=True)