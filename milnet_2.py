'''
Model: Multiple Instance Learning Network
Data: IMDB data from http://ai.stanford.edu/~amaas/data/sentiment/ 
Embedding Weights: GLOVE from http://nlp.stanford.edu/data/glove.6B.zip

Author: Yiqing (Louis) Luo
Reference: Angelidis and Lapata 2017 ACL Conference
'''

## Data Loading
import sys, os
sys.path.append(os.pardir)

os.environ["KERAS_BACKEND"]='tensorflow'
import numpy as np

x_train = np.load('/home/louis/SharedWindows/x_train_sort.npy')
x_test = np.load('/home/louis/SharedWindows/x_test_sort.npy')
y_train = np.load('/home/louis/SharedWindows/t_train.npy')
y_test = np.load('/home/louis/SharedWindows/t_test.npy')

embWeights=np.load('/home/louis/SharedWindows/weights.npy')
idx=np.load('/home/louis/SharedWindows/index.npy')
embWeights = embWeights[idx]

print('data loaded')

## Import and Initialization

from keras.layers import Input, merge
from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import keras.backend as K
from keras.layers import Lambda, regularizers, Average
from keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, GlobalMaxPooling2D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.layers import merge, Layer, Activation, Dot, Concatenate, Flatten, Lambda
from keras.initializers import Identity,glorot_normal
from keras import regularizers
from keras import metrics
from keras.utils import plot_model

numSentencesPerDoc, numWordsPerSentence = x_train[0].shape[0], x_train[0].shape[1]
print("Number of sentences and words:")
print(numSentencesPerDoc, numWordsPerSentence)

vocabSize, embeddingSize = embWeights.shape[0], embWeights.shape[1]
print(vocabSize, embeddingSize)

#Hyperparameters
filters = 1 
windowMin = 2
windowMax = 6
batch_size = 256
epochs = 25
numGRU = 100
numDensePool=50
dr= 0.5

## Layer Declaration

x_in = Input( shape = ( numSentencesPerDoc, numWordsPerSentence ) , name='Input' )
embLayer = Embedding( input_dim=embWeights.shape[0], output_dim=embWeights.shape[1], weights=[embWeights]
                      ,mask_zero=True , trainable=True, embeddings_regularizer=regularizers.l2(0.0000001)
                      , input_length=numWordsPerSentence, name='Embedding' )

maxPooledPerDoc = []
convNets = []
maxPools = []

extraDimLayer = Lambda(lambda x: K.expand_dims(x), name='extraDimForConvo')
squeezeThirdLayer = Lambda(lambda x: K.squeeze(x, 3), name='squeezeThirdLayer')

for windowSize in range(windowMin,windowMax):
    name='word_mat_convo_win_size_'+str(windowSize)
    convNet = Conv2D(filters, kernel_size=(windowSize,embeddingSize), padding='valid', 
                           activation='relu', strides=1, use_bias=True, input_shape=(numWordsPerSentence, embeddingSize, 1), data_format="channels_last",
                           kernel_initializer=glorot_normal(),kernel_regularizer=regularizers.l2(),name=name)
    convNets.append(convNet)
    name='word_mat_max_pool_win_size_'+str(windowSize)
    maxPool = MaxPooling1D(pool_size = int(numWordsPerSentence-windowSize-1), padding='valid')
    maxPools.append(maxPool)
    
    
for i in range(numSentencesPerDoc):
    maxPooledPerSentence = []
    x_pop = Lambda(lambda x: x[:,i], output_shape=(numWordsPerSentence, ) , name='convert_shape_'+'sentence'+str(i+1))( x_in )

    for j in range(windowMax-windowMin):   
        emb = embLayer(x_pop)
        emb = Dropout(dr)(emb)
        reshaped = extraDimLayer(emb)
        name='word_mat_convo_win_size_'+str(j)+'_sentence_'+str(i)

        wordsCNN  = convNets[j](reshaped)
        wordsCNN=Dropout(dr)(wordsCNN)
        squeezed = squeezeThirdLayer(wordsCNN)
        wordsCNNPooled=GlobalMaxPooling1D()(squeezed)
        maxPooledPerSentence.append(wordsCNNPooled)
        
    mergedPoolForSentence = Concatenate(axis = 1)(maxPooledPerSentence)
    newShape=(-1,1,int(mergedPoolForSentence.shape[1]))
    reshapedPoolForSentence = Lambda(lambda x: K.reshape(x,shape=newShape), name ='switch_axis_'+'sentence'+str(i+1)+'winSize'+str(j+windowMin))(mergedPoolForSentence)
    densePoolForSentence = Dense(numDensePool, activation='softmax', use_bias=True)(reshapedPoolForSentence)

    maxPooledPerDoc.append(densePoolForSentence)
    
#Naive (Average) Approach
averaged = Average()(maxPooledPerDoc) 
averaged = Lambda(lambda x:K.reshape(x,shape=(-1,int(averaged.shape[1])*int(averaged.shape[2]))), name ='attend_output')(averaged)
out_avg = Dense(1, activation='sigmoid', use_bias=True)(averaged) 
    
#Apply Attention 
mergedPoolPerDoc = Concatenate(axis = 1)(maxPooledPerDoc)
biRnn_ = Bidirectional(GRU(int(mergedPoolPerDoc.shape[2]),  return_sequences=True), merge_mode='concat')(mergedPoolPerDoc)
newShape = (-1, int(mergedPoolPerDoc.shape[1]), 2*int(mergedPoolPerDoc.shape[2]))
biRnn = Lambda(lambda x: K.reshape(x,shape=newShape), name ='biRnn_TF_Reminder')(biRnn_)

CONTEXT_DIM = 100

eij = Dense(CONTEXT_DIM, use_bias=True, activation='tanh')(biRnn)
eij = Dense(CONTEXT_DIM, use_bias=False, activation='softmax')(eij)

weighted_input_ = Dot(axes = 1)([eij, biRnn])
weighted_input = Lambda(lambda x: K.reshape(x,shape=(-1,int(weighted_input_.shape[1])*int(weighted_input_.shape[2]))), name ='attend_output')(weighted_input_)

out = Dense(1, activation='sigmoid', use_bias=True)(weighted_input)




## Model with Attention

model = Model(input=[x_in], output=[out])
# adadelta = keras.optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=None, decay=0.0)
# model.compile(loss='binary_crossentropy',
#               optimizer=adadelta,
#               metrics=['accuracy'])
              
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

print("Attention Model Build Complete")

## Model without Attention
model_avg = Model(inputs=[x_in], outputs=[out_avg])
model_avg.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

print("Average Model Build Complete")

## Save model to png file
from keras.utils import plot_model
plot_model( model, to_file='model.png' )

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG( model_to_dot( model ).create( prog='dot', format='svg' ) )

## Training
print('Train...')
history = model.fit(x_train, y_train, batch_size = batch_size, verbose=1, epochs=epochs
                    ,validation_split=0.2, shuffle=True)
                    
                    
                    