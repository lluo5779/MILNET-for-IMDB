import numpy as np
import keras.preprocessing.text as kpt
import re
import sys, os
import MeCab
sys.path.append(os.pardir)
from keras.preprocessing import sequence

################################### class ###########################################

class TextDeal( ):
    def __init__(self):
        self.max_items = 20   #文のかっと
        self.max_length = 25  #文の長さのかっと
        self.zero_array = list( np.array([np.zeros( self.max_length )], dtype = 'int64') )
        
    def data2text( self, num, data, save_path ):
        if num < 10:
            filename = 'im000000' + str(num) + '.txt'
        elif num >= 10 and num < 100:
            filename = 'im00000' + str(num) + '.txt'
        elif num >= 100 and num < 1000:
            filename = 'im0000' + str(num) + '.txt'
        elif num >= 1000 and num < 10000:
            filename = 'im000' + str(num) + '.txt'
        elif num >= 10000 and num < 100000:
            filename = 'im00' + str(num) + '.txt'
        elif num >= 100000 and num < 1000000:
            filename = 'im0' + str(num) + '.txt'
        else :
            filename = 'im' + str(num) + '.txt'

        path = save_path + filename #save path
        f = open( path, 'w', encoding = 'utf8' )

        text = ''
        error_word = '\ufeff'
        for idx in range( 1, 60+1 ):
            sentence = str( data[idx] )
            if sentence == 'nan':
                continue
            else:
                if sentence[0] == '\ufeff':
                    text += (sentence[1:] + '\n') #1st char of 1st sentence is '\ufeff'
                else:
                    text += (sentence + '\n')

        f.write( text )
        f.close()


    def wakati( self, load_path, save_path, name ):
        
        f = open( load_path, 'r', encoding = 'utf8' )
        text = f.readlines()
        f.close()
        
        mc = MeCab.Tagger( '-Owakati' )
        result = ''
        for i in range( len( text ) ):
            result += mc.parse( text[i] )
            
        path = save_path + name #保存するパスの設定
        g = open( path, 'w', encoding = 'utf8' )
        g.writelines( result )
        g.close()
        del( mc )
        
    def text_replace( self, load_path, save_path, name ):
        f = open( load_path, 'r', encoding = 'utf8' )
        text = f.readlines()
        f.close()
        
        result = ''
        for i in range( len(text) ):
            tmp = ' ' + text[i]
            tmp = re.sub( r'？？+', '？', tmp )
            tmp = re.sub( r'！！+', '！', tmp )
            tmp = re.sub( r'、、+', '、', tmp )
            tmp = re.sub( r'。。+', '。', tmp )
            tmp = re.sub( '”|＃|＄|％|（|）|＊|＋|、|・|：|；|＜|＝|＞|＠|「|￥|」|＾|＿|｀|｛|｜|｝|〜', ' ', tmp ) #ー
            tmp = re.sub( '。|？|！', '\n' ,tmp)
            tmp = tmp.replace( '\n', 'fffff' )
            tmp = re.sub( r'fffff fffff+', '\n', tmp )
            tmp = tmp.replace( 'fffff', '\n' )
            tmp = re.sub( r'  +', ' ', tmp )
            tmp = tmp.replace( '\n\n', '\n' )
            
            result += tmp
        
        path = save_path + name #保存するパスの設定
        g = open( path, 'w', encoding = 'utf8' )
        g.write( result )
        g.close()

        
    def text2index( self, vocab, load_path, save_path, filename ):
        f = open( load_path, 'r', encoding = "utf-8" ) #文字ファイルの読み込み
        tmp = f.readlines()
        f.close()

        for i in range( len( tmp ) ):
            tmp[i] = list( filter( lambda x: x in vocab, tmp[i].split() ) )
            tmp[i] = np.array( list( map( lambda x : vocab[x].index, tmp[i] ) ) )

        mat = []
        cnt = 0
        for s in tmp:
            if len(s)==0 : continue
            if len(s)<=2 and cnt>=1 :
                mat[cnt-1] = np.append( mat[cnt-1], s, axis=0 )
                continue
            if cnt>=1 and len(mat[cnt-1])<=2 :
                mat[cnt-1] = np.append( mat[cnt-1], s, axis=0 )
                continue
            mat.append( s )
            cnt = cnt+1
            
        myarray = np.asarray( mat )
        l = list( map(lambda x: len(x), myarray) )
        len_avg = np.mean( l )
        len_min = np.min( l )
        len_max = np.max( l )
        np.save( save_path + filename + '.npy', myarray )
        return( [len_avg,len_min,len_max] )
    
    
    def text2index_nishimoto( self, vocab, load_path, save_path, filename ):
        f = open( load_path, 'r', encoding = "utf-8" ) #文字ファイルの読み込み
        tmp = f.readlines()
        f.close()

        for i in range( len( tmp ) ):
            tmp[i] = list( filter( lambda x: x in vocab, tmp[i].split() ) )
            tmp[i] = np.array( list( map( lambda x : vocab.index(x), tmp[i] ) ) )

        myarray = np.asarray( tmp )
        np.save( save_path + filename + '.npy', myarray )

    
    
    def padding( self, load_path, save_path, filename ):
        mat = np.load( load_path )
        mat_len = len( mat )
        
        '''文をある単語数以降カットしたいときに使う。
        　 今は全単語（ self.max_length=50 ）を使っている。
        '''
        self.zero_array = list( np.array([np.zeros( self.max_length )], dtype = 'int64') )
        
        myarray = []
        for i in range( mat_len ):
            if len( mat[i] ) > self.max_length:
                myarray.append( mat[i][:self.max_length] )
            else:
                myarray.append( mat[i] )
        del( mat )
        
        
        
        # Padding about sequence length
        mat_pad = sequence.pad_sequences( myarray, maxlen=self.max_length )
        
        
        # Padding about number of sequences

        self.max_items=18
        if mat_len > self.max_items:
            np.save( save_path + filename, mat_pad[ :self.max_items ] )
            return 0

        tmp = []
        for i in range( self.max_items ):
            if i < mat_len:
                tmp.extend( list( [mat_pad[i]] ) )
            elif i >= mat_len:
                tmp.extend( self.zero_array )

        tmp = np.array( tmp )
        np.save( save_path + filename, tmp )
        return 0

    
    def padding_plus1( self, load_path, save_path, filename ):
        mat = np.load( load_path )
        mat_len = len( mat )
        
        self.zero_array = list( np.array([np.zeros( self.max_length )], dtype = 'int64') )
        
        myarray = []
        for i in range( mat_len ):
            if len( mat[i] ) > self.max_length:
                myarray.append( mat[i][:self.max_length] )
            else:
                myarray.append( mat[i] )
        del( mat )
        
        
        
        # Padding about sequence length
        myarray = np.array( myarray ) + 1
        mat_pad = sequence.pad_sequences( myarray, maxlen=self.max_length )
        
        
        # Padding about number of sequences

        if mat_len > self.max_items:
            np.save( save_path + filename, mat_pad[ :self.max_items ] )
            return 0

        tmp = []
        for i in range( self.max_items ):
            if i < mat_len:
                tmp.extend( list( [mat_pad[i]] ) )
            elif i >= mat_len:
                tmp.extend( self.zero_array )

        tmp = np.array( tmp )
        np.save( save_path + filename, tmp )
        return 0        
        
        
#########################################################################################


    
    
def text_replace(sentences):
    tmp = []
    tmp = re.sub(r'<br />*', ' ', sentences)
    tmp = tmp.replace('\t','')
    tmp = tmp.replace('\n','')
    tmp = tmp.replace('.','\n')
    tmp = re.sub(r'\n\n+', '', tmp)
    tmp = tmp.replace('&', 'and')
    
    return(list(kpt.text_to_word_sequence(tmp, filters='!"#$%()*+,-/:;<=>?@[\\]^_`{|}~', lower=True, split=" ")) )

def neg_modify(sentences, num):
    if num < 10:
        filename = '0000' + str(num) + '_tr_neg.txt'
    elif num >= 10 and num < 100:
        filename = '000' + str(num) + '_tr_neg.txt'
    elif num >= 100 and num < 1000:
        filename = '00' + str(num) + '_tr_neg.txt'
    elif num >= 1000 and num < 10000:
        filename = '0' + str(num) + '_tr_neg.txt'
    else :
        filename = str(num) + '_tr_neg.txt'
        
    path_filename = './tr_data/neg/' + filename
    f = open(path_filename, 'w', encoding = 'utf8')
    text = ' '
    text += ' '.join(sentences)
    f.write(text)
    f.close()

def neg_modify_for_test(sentences, num):
    if num < 10:
        filename = '0000' + str(num) + '_te_neg.txt'
    elif num >= 10 and num < 100:
        filename = '000' + str(num) + '_te_neg.txt'
    elif num >= 100 and num < 1000:
        filename = '00' + str(num) + '_te_neg.txt'
    elif num >= 1000 and num < 10000:
        filename = '0' + str(num) + '_te_neg.txt'
    else :
        filename = str(num) + '_te_neg.txt'
        
    path_filename = './te_data/neg/' + filename
    f = open(path_filename, 'w', encoding = 'utf8')
    text = ' '
    text += ' '.join(sentences)
    f.write(text)
    f.close()


def pos_modify(sentences, num):
    if num < 10:
        filename = '0000' + str(num) + '_tr_pos.txt'
    elif num >= 10 and num < 100:
        filename = '000' + str(num) + '_tr_pos.txt'
    elif num >= 100 and num < 1000:
        filename = '00' + str(num) + '_tr_pos.txt'
    elif num >= 1000 and num < 10000:
        filename = '0' + str(num) + '_tr_pos.txt'
    else :
        filename = str(num) + '_tr_pos.txt'
        
    path_filename = './tr_data/pos/' + filename
    f = open(path_filename, 'w', encoding = 'utf8')
    text = ' '
    text += ' '.join(sentences)
    f.write(text)
    f.close()
    
    
def pos_modify_for_test(sentences, num):
    if num < 10:
        filename = '0000' + str(num) + '_te_pos.txt'
    elif num >= 10 and num < 100:
        filename = '000' + str(num) + '_te_pos.txt'
    elif num >= 100 and num < 1000:
        filename = '00' + str(num) + '_te_pos.txt'
    elif num >= 1000 and num < 10000:
        filename = '0' + str(num) + '_te_pos.txt'
    else :
        filename = str(num) + '_te_pos.txt'
        
    path_filename = './te_data/pos/' + filename
    f = open(path_filename, 'w', encoding = 'utf8')
    text = ' '
    text += ' '.join(sentences)
    f.write(text)
    f.close()

def wrapper_func(tuple_data):
    return(tuple_data[0](tuple_data[1],tuple_data[2]) )

def wrapper_func_three(tuple_data):
    return(tuple_data[0](tuple_data[1],tuple_data[2],tuple_data[3]) )

def wrapper_func_four(tuple_data):
    return(tuple_data[0](tuple_data[1],tuple_data[2],tuple_data[3],tuple_data[4]) )



def sentence2index(vocab, before_path, after_path, filename):
    f = open(before_path + filename + '.txt', 'r', encoding = "utf-8")#文字ファイルの読み込み
    tmp = f.readlines()
    f.close()
    max_len = 0
    for i in range(len(tmp)):
        if len(tmp[i]) > max_len:
            max_len = len(tmp[i])
        tmp[i] = list(filter(lambda x: x in vocab, tmp[i].split()))
        tmp[i] = list(map(lambda x : vocab[x].index, tmp[i]))
        
    myarray = np.asarray(tmp)
    np.save(after_path + filename + '.npy', myarray)
    
    return(max_len)


    
from keras.preprocessing import sequence
'''
文長の長さの中央値がだいたい200
各文長350以下の文章はだいたい86%くらい
これ以上長い文章は切り捨てするがこのとき、1つの文章だけなくなる。
'''

def index2padding(before_path, after_path, filename):
    maxlen = 50 #padding_size   #350
    tmp = np.load(before_path + filename + '.npy')
    myarray = []
    for i in range(len(tmp)):
        if len(tmp[i]) > maxlen:
            continue
        myarray.append(tmp[i])
    if len(myarray) == 0:
        return(0)
    del(tmp)
    pad_array = sequence.pad_sequences(myarray, maxlen=maxlen)
    del(myarray)
    np.save(after_path + filename + '.npy', pad_array)
    return(len(pad_array))
    



def load_npy(path):
    return(np.load(path))



def padding_mat(x):
    length = len(x)
    maxlen = 10  #120
    if length > maxlen:
        return( x[:maxlen] )
    
    tmp = []
    add_array = list( np.array([np.zeros(50)], dtype = 'int64') )
    for i in range(maxlen):
        if i < length:
            tmp.extend( list( [x[i]] ) )
        elif i >= length:
            tmp.extend( add_array )
            
    return np.array(tmp)


















    
    
    
    
    

