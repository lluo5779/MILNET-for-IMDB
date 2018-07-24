import sys, os
sys.path.append(os.pardir)

os.environ["KERAS_BACKEND"]='tensorflow'
#import glob
import numpy as np

x_train = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_features.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/x_train_sort.npy')
# y_train = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_scores_binary.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/x_test_sort.npy')
# print(len(x_train))
# 
x_test = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_features.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/t_train.npy')
y_test = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_scores_binary.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/t_test.npy')
# 
# x_valid = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_features.npy')
# y_valid = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_scores_binary.npy')
# 
train_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_idx_binary_only.npy')
# test_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_idx_binary_only.npy')
# valid_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_idx_binary_only.npy')

#print(len(train_idx))
x_train = x_train[train_idx]
#y_train = y_train[train_idx]
# print(len(x_train))
# print(len(y_train))

# x_test = x_test[test_idx]
# #y_test = y_test[test_idx]
# 
# x_valid = x_valid[valid_idx]
#y_valid = y
#_valid[valid_idx]
#print(len(x_train))

# word_idx=np.load( '/home/louis/SharedWindows/edu_data/Preprocessed/' + 'vocab_idx.npy')
#print(idx)
embWeights=np.load( '/home/louis/SharedWindows/edu_data/Preprocessed/' + 'weights.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/weights.npy')

#print(len(word_idx))
# print(y_test)


print('data loaded')

##

import keras
from keras.layers import Input, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import keras.backend as K
from keras.layers import Lambda, regularizers, Average

from keras.layers import RepeatVector, Permute, Multiply

from keras.models import Model
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

##
numSentencesPerDoc, numWordsPerSentence = x_train[0].shape[0], x_train[0].shape[1]
print(numSentencesPerDoc, numWordsPerSentence)
#print(x_train[0])

vocabSize, embeddingSize = embWeights.shape[0], embWeights.shape[1]
print(vocabSize, embeddingSize)

dropWordEmb = 0.25
recursiveClass = GRU

filters = 100 #embeddingSize*2
windowMin = 3
windowMax = 6# dimOfSentimentMetrics = 5
batch_size = 200
#epochs = 25
dimGRU = 50
numDensePool=10
eta = 1e-4
dr = 0.5

##

#wordsInputs = Input(shape=(numWordsPerSentence,1), batch_shape=(numSentencesPerDoc,numWordsPerSentence,), dtype='int32', name='words_input')

x_in = Input( shape = ( numSentencesPerDoc, numWordsPerSentence ) , name='Input' )
#x_pop = Lambda( lambda x: x, output_shape=(numWordsPerSentence, ) , name='convert_shape' )( x_in )
    
#Layer functionの定義
embLayer = Embedding( input_dim=embWeights.shape[0], output_dim=embWeights.shape[1], weights=[embWeights]
                      ,mask_zero=False , trainable=False, embeddings_regularizer=regularizers.l2(0.0000001)
                      , input_length=numWordsPerSentence, name='Embedding' )


maxPooledPerDoc = []
convNets = []
maxPools = []

extraDimLayer = Lambda(lambda x: K.expand_dims(x), name='extraDimForConvo')
squeezeThirdLayer = Lambda(lambda x: K.squeeze(x, 3), name='squeezeThirdLayer')

for windowSize in range(windowMin,windowMax):
    name='word_mat_convo_win_size_'+str(windowSize)
    #convNet = Conv2D(filters, kernel_size=(windowSize,embeddingSize), padding='valid', activation='relu'
    #                 ,strides=1, use_bias=True, input_shape=(numWordsPerSentence, embeddingSize, 1), data_format="channels_last",kernel_initializer=glorot_normal()
    #                 ,bias_regularizer=regularizers.l2(eta), kernel_regularizer=regularizers.l2(eta),name=name)
    convNet = Conv1D(filters=filters, kernel_size=windowSize, padding='valid', activation='relu', strides=1, name=name)
    convNets.append(convNet)
    name='word_mat_max_pool_win_size_'+str(windowSize)
    maxPool = MaxPooling1D(pool_size = int(numWordsPerSentence-windowSize-1), padding='valid')
    maxPools.append(maxPool)
    
    
for i in range(numSentencesPerDoc):
    maxPooledPerSentence = []
    x_pop = Lambda(lambda x: x[:,i], output_shape=(numWordsPerSentence, ) , name='convert_shape_'+'sentence'+str(i+1))( x_in )

    for j in range(windowMax-windowMin):   
        emb = embLayer(x_pop)
        #emb = Dropout(dr,name='DropEmb'+str(i)+str(j))(emb)
        #reshaped = extraDimLayer(emb)#Lambda(lambda x: K.expand_dims(x), name='extraDimForConvo_'+str(j)+'_sentence_'+str(i))(emb)
        #name='word_mat_convo_win_size_'+str(j)+'_sentence_'+str(i)
        #wordsCNN = Conv2D(filters, kernel_size=(windowSize,embeddingSize), padding='valid', 
        #                    activation='relu', strides=1, use_bias=True, input_shape=(numWordsPerSentence, embeddingSize, 1), data_format="channels_last",
        #                    kernel_initializer=glorot_normal(),kernel_regularizer=regularizers.l2(),name=name)(reshaped)
        wordsCNN  = convNets[j](emb)
        #wordsCNN = Dropout(dr,name='DropCNN'+str(i)+str(j))(wordsCNN)
        #squeezed = squeezeThirdLayer(wordsCNN)#Lambda(lambda x: K.squeeze(x, 3), name='squeezeThirdLayer_'+str(j)+'_sentence_'+str(i))(wordsCNN)
        # newShape = (-1, int(squeezed.shape[1])*int(squeezed.shape[2]))
        # squeezed = Lambda(lambda x: K.reshape(x,shape=newShape), name ='squeezeDimForMaxPool'+str(i)+str(j))(squeezed)
        #wordsCNNPooled=GlobalMaxPooling1D()(squeezed)
        #wordsCNNPooled= MaxPooling1D(pool_size = int(squeezed.shape[1]), padding='valid')(squeezed)
        wordsCNNPooled = MaxPooling1D(pool_size=(numWordsPerSentence-(j+windowMin)+1))(wordsCNN)
        flattened = Lambda(lambda x: K.squeeze(x, 1))(wordsCNNPooled)
        maxPooledPerSentence.append(flattened)
        
    mergedPoolForSentence = Concatenate(axis = 1)(maxPooledPerSentence)
    newShape=(-1,1,int(mergedPoolForSentence.shape[1]))
    reshapedPoolForSentence = Lambda(lambda x: K.reshape(x,shape=newShape), name ='switch_axis_'+'sentence'+str(i+1)+'winSize'+str(j+windowMin))(mergedPoolForSentence)
    densePoolForSentence = Dense(numDensePool, bias_regularizer=regularizers.l2(eta),
                                 kernel_regularizer=regularizers.l2(eta), activation='softmax', use_bias=True)(reshapedPoolForSentence)

    densePoolForSentence = Dropout(dr,name='DropDense'+str(i))(densePoolForSentence)
    maxPooledPerDoc.append(densePoolForSentence)
    
#Naive Approach
averaged = Average()(maxPooledPerDoc) 
averaged = Lambda(lambda x:K.reshape(x,shape=(-1,int(averaged.shape[1])*int(averaged.shape[2]))), name ='attend_output')(averaged)
out_avg = Dense(1, activation='sigmoid', use_bias=True)(averaged) 
    
#Apply Attention 
mergedPoolPerDoc = Concatenate(axis = 1)(maxPooledPerDoc)
biRnn_ = Bidirectional(GRU(dimGRU,  return_sequences=True), merge_mode='concat')(mergedPoolPerDoc)
newShape = (-1, int(mergedPoolPerDoc.shape[1]), int(biRnn_.shape[2]))
biRnn = Lambda(lambda x: K.reshape(x,shape=newShape), name ='biRnn_TF_Reminder1')(biRnn_)
#biRnn2 = Lambda(lambda x: K.reshape(x,shape=newShape), name ='biRnn_TF_Reminder2')(biRnn_[1])

#QIITA
#repeat_dec = TimeDistributed(RepeatVector(numSentencesPerDoc), name='repeat_')(biRnn)
#annotation_layer = TimeDistributed(Dense(CONTEXT_DIM))(biRnn)



#biRnn_cat = Concatenate(axis = 2)([biRnn1, biRnn2])

CONTEXT_DIM = 100

eij_ = Dense(CONTEXT_DIM, use_bias=True, activation='tanh')(biRnn)
eij = Dense(1, use_bias=False, name = 'attention_weights')(eij_)
eij_permuted = Permute((2,1))(eij)
eij_normalized = Activation('softmax',name = 'attention_softmax')(eij_permuted)
eij_normalized = Permute((2,1))(eij_normalized)

weighted_input = Dot(axes=1,name='attention_mult')([mergedPoolPerDoc,eij_normalized])


weighted_input = Dot(axes = 1)([eij, mergedPoolPerDoc])
#weighted_input = Lambda(lambda x: K.reshape(x,shape=(-1,int(weighted_input_.shape[1])*int(weighted_input_.shape[2]))), name ='attend_output')(weighted_input_)
weighted_input = Lambda(lambda x: K.squeeze(x, 1), name='squeezeOutput')(weighted_input)

out = Dense(1, activation='sigmoid', use_bias=True)(weighted_input)




##

model = Model(inputs=[x_in], outputs=[out, eij, eij_,eij_normalized, mergedPoolPerDoc])
#adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#model.compile(loss='binary_crossentropy',
#              optimizer=adadelta,
#              metrics=['accuracy'])
         
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

print("Attention Model Build Complete")
##
weight_path = '/home/louis/SharedWindows/results/params_milnet_adam_0719_2_adjCNN_fixed_weights.hdf5'
model.load_weights(weight_path)


##
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_weights(w):
    l = len(w)
    fig = plt.figure(figsize=(30,16))
    gs = gridspec.GridSpec(int(l/2+1),2)
    
    for i in range(l):
        j = int( i/2 )
        k = i%2
        if k==0:
            ax = plt.subplot(gs[j,k])
            ax.hist( w[i].flatten() ,bins=100 ,range=(-0.5,0.5) )
            #ax.set_title(title[i])
        else:
            ax = plt.subplot(gs[j,k])
            ax.hist( w[i].flatten() ,bins=100 ,range=(-0.5,0.5) )
            #ax.set_title(title[i])
    plt.show()
    #plt.close()

##
model.summary()
##
plot_weights(model.get_layer('word_mat_convo_win_size_4').get_weights())
##
#print(x_test[0])
test = x_test[1:3].tolist()
for l in range(len(test)):
    for i in range(len(test[l])):
        for j in range(len(test[l][i])):
            if test[l][i][j] >= 61560:
                test[l][i][j] = 0

outputs = model.predict_on_batch(np.asarray(test))
##
import time

layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'word_mat_convo_win_size_3'
filter_index = 0
input_img = model.input

for filter_index in range(100):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    layer_output = layer_dict[layer_name].get_output_at(0)
    loss = K.mean(layer_output[:,:,filter_index])#, axis=[0,1,2])
    
    
    grads = K.gradients(loss, input_img)[0]

    iterate = K.function([input_img], [loss, grads])

    step = 1.0

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, int(input_img[1]), int(input_img[2])))
    
    input_img_data = input_img_data * 10000
    
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
    
##
from __future__ import print_function

import numpy as np
import time

from keras.applications import vgg16
from keras import backend as K
img_width = 128
img_height = 128

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'block5_conv1'

# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# build the VGG16 network with ImageNet weights
model = vgg16.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


kept_filters = []
for filter_index in range(200):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

