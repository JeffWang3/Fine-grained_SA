#-*- coding: utf-8 -*-
import re
import sys
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

validate_data_path = "/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"
train_data_path = "/data1/hjw/fine_grit_emotion_analysis/train/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"
#load the data
def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df

#train = load_data_from(train_data_path)
data = load_data_from_csv(validate_data_path)

def rmitems(text):
    '''
    remove useless items
    '''
    remove_items = u'[a-zA-Z0-9０-９Ａ-Ｙａ-ｙ，。？：“”"＝/#＠@§；,.:-?;\'&()<>＋（）_《》\uf8e7\u3000\x0c\／^L·-、、．%\%％－[\]［ ］\x7f-]+'
    text_rmitems = re.sub(remove_items, '', text)
    text_rmblank = re.sub("\t", '',text_rmitems)
    return text_rmblank

y_cols = ['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find','service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed','price_level','price_cost_effective','price_discount','environment_decoration','environment_noise','environment_space','environment_cleaness','dish_portion','dish_taste','dish_look','dish_recommendation','others_overall_experience','others_willing_to_consume_again']

sentence = []
for i in range(len(data)):
    sentence.append(rmitems(data['content'][i].replace("\n","")))

sentence = pd.Series(sentence)
sentence = sentence.apply(lambda x:list(jieba.cut(x)))

for index,outf in enumerate(y_cols):
    outfile = "/data1/hjw/train/"+outf+'.csv'
    fout = open(outfile,'w')
    label = data[y_cols[index]]+2
    for i in range(len(data)):
        pdb.set_trace()
        fout.write(" ".join(sentence[i]).encode('utf-8')+"\t__label__"+str(label[i])+"\n")
    fout.close()

