import numpy as np
from pyfasttext import FastText
import pandas as pd
import jieba
import pdb
#data = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")
#data = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv",header=0,encoding="utf-8")
#data = data[0:10]
#data['words'] = data['content'].apply(lambda x:list(jieba.cut(x)))
#data['words'] = data['words'].apply(lambda x:x[1:-1])

#train = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/train/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv",header=0,encoding="utf-8")
#data = data[0:10]
#train['words'] = train['content'].apply(lambda x:list(jieba.cut(x)))
#train['words'] = train['words'].apply(lambda x:x[1:-1])

test = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/testA/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv",header=0,encoding="utf-8")
#data = data[0:10]
test['words'] = test['content'].apply(lambda x:list(jieba.cut(x)))
test['words'] = test['words'].apply(lambda x:x[1:-1])

y_cols = ['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find','service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed','price_level','price_cost_effective','price_discount','environment_decoration','environment_noise','environment_space','environment_cleaness','dish_portion','dish_taste','dish_look','dish_recommendation','others_overall_experience','others_willing_to_consume_again']

#produce the sentence matrix
data_dm = 300
input1 = []
model = FastText('/data1/hjw/fine_grit_emotion_analysis/vec/data/vec3.bin')

#once deal with one sentence
def produce_input(sentence):
    matrix1 = []
    dim = min(300,len(sentence))
    for i in range(dim):
        matrix1.append(model[sentence[i]])
    if dim < 300:
        for i in range(dim,300):
            matrix1.append(range(192))
    mat = np.array(matrix1)
#    print(np.shape(mat))
#    pdb.set_trace()
    return mat.reshape(300,192)

def produce_inputf(sentence_batch):
    input2 = []
    for i in range(len(sentence_batch)):
        input2.append(produce_input(sentence_batch[i]))
    return input2

train_x = produce_inputf(test['words'])
train_x = np.array(train_x)
np.save("test1.npy",train_x)
