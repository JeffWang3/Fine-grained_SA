#-*- coding: utf-8 -*-
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
import keras as ks
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import layers
import pdb
import logging
from pyfasttext import FastText
import re

vec = FastText('/data1/hjw/cc.zh.300.bin')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

model_save_path = os.path.abspath('..') + "/data/"
train_data_path = "/data1/hjw/fine_grit_emotion_analysis/train/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"
validate_data_path = "/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"
test_data_path = "/data1/hjw/fine_grit_emotion_analysis/testA/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"
test_data_predict_out_path = "/data1/hjw/fine_grit_emotion_analysis/test_prediction/"
stop_path = "/data1/hjw/stop_words.txt"

l1 = [0,3,7,10,14,18]

#load the data
def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df

def load_stopword(file_name):
    list1 = []
    with open(file_name) as fn:
        for word in fn.readlines():
            list1.append(word)
        return list1

def rmitems(text):
    '''
    remove useless items
    '''
    remove_items = u'[a-zA-Z0-9０-９Ａ-Ｙａ-ｙ，。？：“”"＝/#＠@§；,.:-?;\'&()<>＋（）_《》\uf8e7\u3000\x0c\／^L·-、、．%\%％－[\]［ ］\x7f-]+'
    text_rmitems = re.sub(remove_items, '', text)
    text_rmblank = re.sub("\t", '',text_rmitems)
    return text_rmblank

#stoplist = []
#with open(stop_path,'r',encoding="utf-8") as f:
#    for word in f.readlines():
#        stoplist.append(word)

train = load_data_from_csv(train_data_path)
val = load_data_from_csv(validate_data_path)
test = load_data_from_csv(test_data_path)

#test1 = test.iloc[:500,:]

data = pd.concat([train,val])
data = pd.concat([data,test])

#data = pd.concat([train.iloc[:1000,:],val.iloc[:500,:]])
#data = pd.concat([data,test.iloc[:500,:]])

y_cols = ['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find','service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed','price_level','price_cost_effective','price_discount','environment_decoration','environment_noise','environment_space','environment_cleaness','dish_portion','dish_taste','dish_look','dish_recommendation','others_overall_experience','others_willing_to_consume_again']

#data_processing

# jieba word processing
data['words'] = data['content'].apply(lambda x:list(jieba.cut(rmitems(x.replace("\n","")))))
# delete the delimeters
data['words'] = data['words'].apply(lambda x:x[1:-1])
# data['len'] = data['words'].apply(lambda x:len(x))
# maxlen = data['len'].max() 
words_dict =[]
texts = []
# remove the stop word
flag = 0

#pdb.set_trace()

for index,row in data.iterrows():
#    pdb.set_trace()
    line = [word for word in list(row['words'])]
    words_dict.extend([word for word in line])
    texts.append(line)
    
#    if flag == 0:
#        print(line)
#        print(words_dict)
#        flag += 1
# stop_data = pd.DataFrame(texts)
# calculate the max_length of every sample
maxlen = 300
#1for line in texts:
#1    if maxlen < len(line):
#1        maxlen = len(line)
max_words=50000                   #50000
##pdb.trace_back()
# utilize Tokenizer of keras to onehot, adjust the unequal-length array
tokenizer = ks.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
data_w = tokenizer.texts_to_sequences(texts)#text to vector
data_T = ks.preprocessing.sequence.pad_sequences(data_w, maxlen=maxlen)
#data redistributed to trainset, testset and validationset
#dealed_train = data_T
dealed_train = data_T[:train.shape[0]]
#dealed_val = data_T[train.shape[0]:(train.shape[0]+100)]
#dealed_test = data_T[(train.shape[0]+100):]
dealed_val = data_T[train.shape[0]:(train.shape[0]+val.shape[0])]
dealed_test = data_T[(train.shape[0]+val.shape[0]):]

#dealed_train = data_T[:1000]
#dealed_val = data_T[1000:1500]
#dealed_test = data_T[1500:]

#build the cnn model
def build_model():
    model = ks.Sequential()
    embedding_dim = 192
    model.add(layers.embeddings.Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(5))
#     model.add(Dropout(0.5))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(5))
#     model.add(Dropout(0.5))
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
#     model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

def train_CV_CNN(train_x=dealed_train, test_x=dealed_test, val_x = dealed_val,y_cols=y_cols, debug=False, folds=2):
#1    model = build_model()
    result = pd.Series()
    
#1    model.compile(optimizer='adam',loss='categorical_crossentropy')
    F1_scores = 0
    F1_score = 0
    if debug:
        y_cols= ['location_traffic_convenience']
    for index,col in enumerate(y_cols):
        model = build_model()
        model.compile(optimizer='adam',loss='categorical_crossentropy')
#        if index in l1:
#            model = build_model()
#            model.compile(optimizer='adam',loss='categorical_crossentropy')
        train_y = train[col]+2
#        train_y = train_y[:1000]
        val_y = val[col]+2
#        val_y = val_y[:500]
#        print(train_y)
#        logger.info(type(val_y))
#        logger.info("----------------------------------------------")
        y_val_pred = 0
        y_test_pred = 0
#         epochs=[5,10]   , stratify=train_y
        for i in range(1):
            X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=100*i,stratify = train_y)

            y_train_onehot = to_categorical(y_train)
            y_test_onehot = to_categorical(y_test)
            history = model.fit(X_train, y_train_onehot, epochs=20, batch_size=64, validation_data=(X_test, y_test_onehot)) #epoch=20
            
            #predict validation and test
            y_val_pred = model.predict(val_x)
#            y_test_pred += model.predict(test_x)
            y_test_pred = model.predict(test_x)
            
            y_val_pred = np.argmax(y_val_pred, axis=1)
#            logger.info("y_val_pred")
#            logger.info(y_val_pred)

            F1_score = f1_score(y_val_pred, val_y, average='macro')
            F1_scores += F1_score

            logger.info(col+' f1_score: '+str(F1_score)+' ACC_score: '+str(accuracy_score(y_val_pred, val_y)))
#        pdb.traceback()
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred-2
        model.save(test_data_predict_out_path+col+'my_model.h5')
#    pdb.traceback()
    logger.info('all F1_score:'+str(F1_scores/len(y_cols)))
    return result

hjw = train_CV_CNN(debug=True)
pdb.set_trace()
#hjw = pd.DataFrame(hjw)
for i in range(len(hjw)):
    for j in range(20):
        test[y_cols[j]] = hjw[y_cols[j]]

test.to_csv('final_result.csv')
