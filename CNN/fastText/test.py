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
#from keras.utils.np_utils import to_categorical
#from keras.models import Sequential
#from keras import layers
import pdb
import logging
from pyfasttext import FastText

outf = "test.txt"
inf = "remove_items.txt"

out = open(outf,'w')
inp = open(inf,'r')

#i = inp.readline()
#print(type(i))
#out.write(inp.readline())


validate_data_path = "/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"
train_data_path = "/data1/hjw/fine_grit_emotion_analysis/train/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"
#load the data
def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df

#train = load_data_from(train_data_path)
data = load_data_from_csv(validate_data_path)

out.write(data['content'][0].encode('utf-8'))

inp.close()
out.close()
