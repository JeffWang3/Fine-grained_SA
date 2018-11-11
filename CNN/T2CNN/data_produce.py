import numpy as np
from pyfasttext import FastText
import pandas as pd
import jieba
import pdb
#data = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")
data = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv",header=0,encoding="utf-8")
#data = data[0:10]
data['words'] = data['content'].apply(lambda x:list(jieba.cut(x)))
data['words'] = data['words'].apply(lambda x:x[1:-1])

#produce the sentence matrix
data_dm = 300
input1 = []
model = FastText('/data1/hjw/cc.zh.300.bin')

#once deal with one sentence
def produce_input(sentence):
    matrix1 = []
    dim = min(300,len(sentence))
    for i in range(dim):
        matrix1.append(model[sentence[i]])
    if dim < 299:
        for i in range(dim,300):
            matrix1.append(range(300))
    mat = np.array(matrix1)
    return mat.reshape(300,300,1)


def produce_inputf(sentence_batch):
    input2 = []
    for i in range(len(sentence_batch)):
        input2.append(produce_input(sentence_batch[i]))
    return input2

input1 = produce_inputf(data['words'])
pdb.set_trace()



#build the cnn model
def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10))

    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

model = build_model()

