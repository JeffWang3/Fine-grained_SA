import fasttext
import sys
import pandas as pd
import pdb
import jieba
import re
from pyfasttext import FastText

def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df

def rmitems(text):
    '''
    remove useless items
    '''
#    remove_items = u'[a-zA-Z0-9０-９Ａ-Ｙａ-ｙ，。？：“”"＝/#＠@§；,.:-?;\'&()<>＋（）_《》\uf8e7\u3000\x0c\／^L·-、、．%\%％－[\]［ ］\x7f-]+'
    remove_items = u'[a-zA-Z0-9０-９Ａ-Ｙａ-ｙ，。？！～~()【】：“”"＝/#＠@§；,.:-?!;\'&()<>＋（）_《》\uf8e7\u3000\x0c\／^L·-、、．%\%％－[\]［ ］\x7f-]+'
    text_rmitems = re.sub(remove_items, '', text)
    text_rmblank = re.sub("\t", '',text_rmitems)
    return text_rmblank

#model = FastText('/data1/hjw/cc.zh.300.bin')
test_data_path = "/data1/hjw/fine_grit_emotion_analysis/testA/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"

y_cols = ['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find','service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed','price_level','price_cost_effective','price_discount','environment_decoration','environment_noise','environment_space','environment_cleaness','dish_portion','dish_taste','dish_look','dish_recommendation','others_overall_experience','others_willing_to_consume_again']

#infile = "/data1/hjw/train/"+y_cols[0]+'.csv'
model_name = y_cols[0]
epoch_num = 20
w2v_u = "/data1/hjw/cc.zh.300.bin"

data = load_data_from_csv(test_data_path)
sen = list(data['content'])

sentence = []
for i in range(len(data)):
    sentence.append(rmitems(data['content'][i].replace("\n","")))

sentence = pd.Series(sentence)
sentence = sentence.apply(lambda x:list(jieba.cut(x)))

sentence1 = []
models = []

for i in range(len(data)):
#    model = FastText('/data1/hjw/cc.zh.300.bin')
    sentence1.append(" ".join(sentence[i]))
for index in y_cols:
    model = FastText('/data1/hjw/cc.zh.300.bin')
    infile = "/data1/hjw/train/"+index+'.csv'
    model.supervised(input=infile,output='/data1/hjw/model/model'+index,epoch=50,lr=0.7)
    models.append(model)
#classifier = fasttext.supervised(infile, model_name, label_prefix='__label__', dim=300, pretrained_vectors=w2v_u, epoch=epoch_num)
#classifier = fasttext.supervised(infile, model_name, label_prefix='__label__')
#pdb.set_trace()
for i in range(20):
    result = models[i].predict_proba(sentence1,k=1)
#pdb.set_trace()
    result = pd.DataFrame(result)
#pdb.set_trace()
    result.to_csv('/data1/hjw/result/result'+str(i)+'.csv')

print('finish\n')

#def load_data_from_csv(file_name, header=0, encoding="utf-8"):

#    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

#    return data_df

#classifier = fasttext.supervised(infile, model_name, label_prefix='__label__', dim=300, pretrained_vectors=w2v_u, epoch=epoch_num)

#classifier = fasttext.supervised(infile, model_name, label_prefix='__label__', dim=300, pretrained_vectors=w2v_u, epoch=epoch_num)
