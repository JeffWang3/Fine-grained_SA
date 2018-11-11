# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:11:18 2018

@author: Jeff
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import jieba
import logging
import argparse
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.externals import joblib
import os
import argparse
import keras
import keras as ks
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras.layers import *
from keras.models import *
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import pdb



model_save_path = os.path.abspath('..') + "/data/"
train_data_path = "./train_data/sentiment_analysis_trainingset.csv"  
validate_data_path = "./train_data/sentiment_analysis_validationset.csv"
test_data_path = "./train_data/sentiment_analysis_testa.csv"
#train_data_path = "./train_data/small_train.csv"  
#validate_data_path = "./train_data/small_val.csv"
#test_data_path = "./train_data/small_test.csv"
test_data_predict_out_path = "./lstm_result"
stop_path = "stop_words.txt"

# load data function
def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df
# load stopwords
def load_stopword(file_name):
    list1 = []
    with open(file_name) as fn:
        for word in fn.readlines():
            list1.append(word)
        return list1

#stoplist = []
#with open(stop_path,'r', encoding='utf-8') as f:
#    for word in f.readlines():
#        stoplist.append(word)

train = load_data_from_csv(train_data_path)
val = load_data_from_csv(validate_data_path)
test = load_data_from_csv(test_data_path)

#concat all datasets
data = pd.concat([train,val])
data = pd.concat([data,test])

y_cols = ['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find','service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed','price_level','price_cost_effective','price_discount','environment_decoration','environment_noise','environment_space','environment_cleaness','dish_portion','dish_taste','dish_look','dish_recommendation','others_overall_experience','others_willing_to_consume_again']

#data_processing

# jieba cut
data['words'] = data['content'].apply(lambda x:list(jieba.cut(x)))
# delete start and ending words
data['words'] = data['words'].apply(lambda x:x[1:-1])
data['count']=data['words'].apply(lambda x:len(x))

words_dict =[]
texts = []

# delete stopwords and create words dictionary
for index,row in data.iterrows():
    #line = [word for word in list(row['words']) if word not in stoplist]
    line = list(row['words'])
    words_dict.extend([word for word in line])
    texts.append(line)
    
#count max legnth of each review
#maxlen = 0
#for line in texts:
#    if maxlen < len(line):
#        maxlen = len(line)
    
max_words=50000
#max_words = 10000 #for small data sets
maxlen = 450

# et index embedding of each review，and pedding rach review into length of maxlen
tokenizer = ks.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
data_w = tokenizer.texts_to_sequences(texts) #convert text into array
data_T = ks.preprocessing.sequence.pad_sequences(data_w, maxlen=maxlen)

# create train, validation, test datasets 
dealed_train = data_T[:train.shape[0]]
dealed_val = data_T[train.shape[0]:(train.shape[0]+val.shape[0])]
dealed_test = data_T[(train.shape[0]+val.shape[0]):]

#pdb.set_trace()

#LSTM without self-attention
def build_basic_model():
    print('Build LSTM model...')
    model = Sequential()
    embedding = 128
    model.add(layers.embeddings.Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))
    model.add(layer.Dense(4, activation='softmax'))
    return model


#build the LSTM model
def build_model():    
    print('Build bi-LSTM self-attention model...') 
    def attention_3d_block(inputs):
        a_probs = Dense(1, activation='softmax')(inputs)
        output = keras.layers.dot([inputs, a_probs], 1, name='attention_mul')
        return output

    # build RNN model with attention
    inputs = Input(shape=(maxlen,))
    embedding_dim = 256
    embeded_words = layers.embeddings.Embedding(max_words, embedding_dim, input_length=maxlen)(inputs)
    lstm_out = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2), name='bilstm')(embeded_words)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    drop2 = Dropout(0.2)(attention_mul)
    output = Dense(4, activation='softmax')(drop2)
    model = Model(inputs=inputs, outputs=output)
    model.summary()
    #pdb.set_trace()
    return model

def train_LSTM(train_x=dealed_train, test_x=dealed_test, val_x = dealed_val,y_cols=y_cols, debug=False, folds=2):
    F1_scores = 0
    F1_score = 0
    #result = pd.Series()
    result = pd.DataFrame()
    result['id'] = test['id']
    result['content'] = test['content']
    #result = test
    #pdb.set_trace()
    
    if debug:
        y_cols= ['location_traffic_convenience']
    for index,col in enumerate(y_cols):
        model = build_model()   
        model.compile(optimizer='adam',loss='categorical_crossentropy')
    
        print("------------------Model No.%d----------------------------"%index)
        train_y = train[col]+2
        val_y = val[col]+2
        y_val_pred = 0
        y_test_pred = 0
        #preprocess train dataset
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=100)
        y_train_onehot = to_categorical(y_train)
        y_test_onehot = to_categorical(y_test)
        #train model
        history = model.fit(X_train, y_train_onehot, epochs=3, batch_size=64, validation_data=(X_test, y_test_onehot))
        # predict on validation dataset
        y_val_pred = model.predict(val_x)
        y_val_pred = np.argmax(y_val_pred, axis=1)
        # count f1 score
        F1_score = f1_score(y_val_pred, val_y, average='macro')
        F1_scores += F1_score
        print(col,'f1_score:',F1_score,'ACC_score:',accuracy_score(y_val_pred, val_y))
        #predict on testA dataset
        y_test_pred += model.predict(test_x)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred-2
        model.save("./lstm_model/"+str(col)+'my_model.h5')
        #pdb.set_trace()
        
    print('all F1_score:',F1_scores/len(y_cols))
    return result

result = train_LSTM()
result.to_csv('./lstm_result/all_result.csv',index=False)
