from pyfasttext import FastText
import pandas as pd
import jieba
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from keras.utils.np_utils import to_categorical
import keras as ks
from keras.models import Sequential,Model
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

train_data_path = "/data1/hjw/fine_grit_emotion_analysis/train/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv"
validate_data_path = "/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv"
test_data_path = "/data1/hjw/fine_grit_emotion_analysis/testA/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"

#data = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")
data = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/validation/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv",header=0,encoding="utf-8")
#data = data[0:10]
#data['words'] = data['content'].apply(lambda x:list(jieba.cut(x)))
#data['words'] = data['words'].apply(lambda x:x[1:-1])

#1train = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/train/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv",header=0,encoding="utf-8")
#data = data[0:10]
#1train['words'] = train['content'].apply(lambda x:list(jieba.cut(x)))
#1train['words'] = train['words'].apply(lambda x:x[1:-1])

#test = pd.read_csv("/data1/hjw/fine_grit_emotion_analysis/testA/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv",header=0,encoding="utf-8")
#data = data[0:10]
#test['words'] = test['content'].apply(lambda x:list(jieba.cut(x)))
#test['words'] = test['words'].apply(lambda x:x[1:-1])

y_cols = ['location_traffic_convenience','location_distance_from_business_district','location_easy_to_find','service_wait_time','service_waiters_attitude','service_parking_convenience','service_serving_speed','price_level','price_cost_effective','price_discount','environment_decoration','environment_noise','environment_space','environment_cleaness','dish_portion','dish_taste','dish_look','dish_recommendation','others_overall_experience','others_willing_to_consume_again']

y_cols = y_cols[:1]

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
    return mat.reshape(300,192)

def produce_inputf(sentence_batch,begin,end):
    input2 = []
    for i in range(begin,end):
        input2.append(produce_input(sentence_batch[i]))
    return input2

def generate_arrays_from_file(path,batch_size,col):
    final = 1500
    index = 0
    end = 0
    f = pd.read_csv(path)
#    final = len(f)
    f=f.iloc[:final,:]
    f['words'] = f['content'].apply(lambda x:list(jieba.cut(x)))
    f['words'] = f['words'].apply(lambda x:x[1:-1])
    while 1:
        if batch_size*index >= final:
            break
        if batch_size*(index+1)>final:
            end = final
        else:
            end = batch_size*(index+1)
        x = produce_inputf(f['words'][batch_size*index:end],begin = batch_size*index, end = end)
        y = f.iloc[batch_size*index:batch_size*(index+1),:][y_cols[col]]
#        pdb.set_trace()
        y = y + 2
#        pdb.set_trace()
        y = to_categorical(y,num_classes=4)
        index = index + 1
#        pdb.set_trace()
        yield(np.array(x),y)

'''
def generate_arrays_from_file2(path,batch_size,col):
    f = pd.read_csv(path)
#    f = f.iloc[:150,:]
    f['words'] = f['content'].apply(lambda x:list(jieba.cut(x)))
    f['words'] = f['words'].apply(lambda x:x[1:-1])
    len1 = len(f)
    indexs = list(range(len1))	
    while 1:
        np.random.shuffle(indexs)
        for i in range(0,len1-batch_size,batch_size):
            x = []
            y = []
            for j in range(i,i+batch_size):
                x.append(produce_input(f['words'][indexs[j]]))
                y.append(f[y_cols[col]][indexs[j]])
#            pdb.set_trace()
            y = np.array(y)
            y = y + 2
            y = to_categorical(y,num_classes=4)
            yield(np.array(x),y)
'''
def generate_arrays_from_file2(path,batch_size,col):
    f = pd.read_csv(path)
#    f = f.iloc[:150,:]
    f['words'] = f['content'].apply(lambda x:list(jieba.cut(x)))
    f['words'] = f['words'].apply(lambda x:x[1:-1])
    len1 = len(f)
    indexs = list(range(len1))
    while 1:
        np.random.shuffle(indexs)
        for i in range(0,len1-batch_size,batch_size):
            x = []
            y = []
            for j in range(i,i+batch_size):
                x.append(produce_input(f['words'][indexs[j]]))
                y.append(f[col][indexs[j]])
#            pdb.set_trace()
            y = np.array(y)
            y = y + 2
            y = to_categorical(y,num_classes=4)
            yield(np.array(x),y)
#######################################################
#val_x = produce_inputf(data['words'],0,15000)
#val_x = np.array(val_x)
#np.save("val1.npy",val_x)
#val_x = np.load("val.npy")


#train_x = produce_inputf(train['words'])
#train_x = np.array(train)
#np.save("train.npy",train_x)
#train_x = np.load("train.npy")

#test_x = produce_inputf(test['words'],0,15000)
#test_x = np.array(test)
#np.save("test1.npy",test_x)

test_x = np.load("test1.npy")
val_x = np.load("val1.npy")

#pdb.set_trace()

#pdb.set_trace()
def build_model():
    model1 = ks.Sequential()
    model1.add(layers.Conv1D(64, 3, activation='relu', input_shape=(300,192)))
    model1.add(layers.MaxPooling1D(5))
#     model.add(Dropout(0.5))
    model1.add(layers.Conv1D(64, 3, activation='relu'))
    model1.add(layers.MaxPooling1D(5))
#     model.add(Dropout(0.5))
    model1.add(layers.Conv1D(64, 3, activation='relu'))
    model1.add(layers.GlobalMaxPooling1D())
#     model.add(layers.Dense(32, activation='relu'))
    model1.add(layers.Dense(4, activation='softmax'))
    return model1

'''
def build_model():
#    model1 = Sequential()
    inputs = layers.Input(shape=(300,300,1),dtype="float32")
    reshape = layers.Reshape((300, 300, 1))(inputs)
    conv_1 = Conv2D(filters=126, kernel_size=(2, 300), activation="relu")(reshape)
    conv_2 = Conv2D(filters=126, kernel_size=(3, 300), activation="relu")(reshape)
    conv_3 = Conv2D(filters=126, kernel_size=(4, 300), activation="relu")(reshape)
#    model1.add(Conv2D(126, (2, 300), input_shape=(300,300,1)))
#    model1.add(Conv2D(126, (3, 300), input_shape=(300,300,1)))
#    model1.add(Activation('relu'))
#    model1.add(MaxPooling2D(pool_size=(5,1)))
#    BatchNormalization()
#    model1.add(Conv2D(32, (3, 1)))
#    model1.add(Activation('relu'))
#    model1.add(MaxPooling2D(pool_size=(5,1)))
#    model1.add(Flatten())
#    model1.add(Dropout(0.5))

    max_1 = MaxPooling2D(pool_size=(300 - 2 + 1, 1), strides=1)(conv_1)
    max_2 = MaxPooling2D(pool_size=(300 - 3 + 1, 1), strides=1)(conv_2)
    max_3 = MaxPooling2D(pool_size=(300 - 4 + 1, 1), strides=1)(conv_3)

#    BatchNormalization()
#    model1.add(Dense(352))
#    model1.add(Activation('relu'))
#    BatchNormalization()
#    model1.add(Dropout(0.2))
#    model1.add(Dense(4))
    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
#    model1.add(Activation('softmax'))
    concat = layers.Concatenate(axis=1)([max_1, max_2, max_3])
    flatten = layers.Flatten()(concat)
    droup_out = layers.Dropout(0.5)(flatten)
    output = layers.Dense(units=4, activation='softmax')(droup_out)

    model1 = Model(inputs=inputs, outputs=output)
    return model1
'''
#build the cnn model
'''
def build_model():
    model1 = ks.Sequential()

    model1.add(Conv2D(32, (3, 3), input_shape=(300,300,1)))
    model1.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model1.add(Conv2D(32, (3, 3)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    BatchNormalization(axis=-1)
    model1.add(Conv2D(64,(3, 3)))
    model1.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model1.add(Conv2D(64, (3, 3)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    model1.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model1.add(Dense(512))
    model1.add(Activation('relu'))
    BatchNormalization()
    model1.add(Dropout(0.2))
    model1.add(Dense(4))
    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    model1.add(Activation('softmax'))
    return model1
'''

#2model1 = build_model()
#2model1.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

#1gen = ImageDataGenerator()
#1test_gen =ImageDataGenerator()

#1pdb.set_trace()

#1train_generator = gen.flow(train_x,train[y_cols[0]],batch_size = 64)
#1test_generator = test_gen.flow(val_x,data[y_cols[0]],batch_size = 64)

#1model1.fit_generator(train_generator,steps_per_epoch=15000//64,epochs=5,validation_data = test_generator,validation_steps = 15000//64)

#model1.fit_generator(generate_arrays_from_file(train_data_path,batch_size = 64,col = 0),steps_per_epoch=105000//64,epochs=5,validation_data = generate_arrays_from_file(validate_data_path,batch_size = 64,col = 0),validation_steps = 15000//64)

#111model1.fit_generator(generate_arrays_from_file(train_data_path,batch_size = 64,col = 0),steps_per_epoch=105000//64,epochs=5)

#2model1.fit_generator(generate_arrays_from_file2(train_data_path,batch_size = 64,col = 0),steps_per_epoch=150//64,epochs=2)

#pdb.set_trace()

#2predictions = model1.predict_classes(test_x)

#2sub = pd.DataFrame({y_cols[0]:predictions})
#2sub.to_csv('./TCNN_pre/'+y_cols[0]+'.csv',index = False)

F1_scores = 0
F1_score = 0

for index,col in enumerate(y_cols):
    model1 = build_model()
    model1.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
#    model1.fit_generator(generate_arrays_from_file2(train_data_path,batch_size = 64,col = 0),steps_per_epoch=105000//64,epochs=2,validation_data = generate_arrays_from_file(validate_data_path,batch_size = 64,col = 0),validation_steps = 15000//64)
#    model1.fit_generator(generate_arrays_from_file2(train_data_path,batch_size = 64,col = index),steps_per_epoch=105000//64,epochs=5)
    model1.fit_generator(generate_arrays_from_file2(train_data_path,batch_size = 64,col = col),steps_per_epoch=105000//64,epochs=2,validation_data=generate_arrays_from_file2(validate_data_path,batch_size = 64,col = col),validation_steps=15000/64)
#    model1.fit_generator(generate_arrays_from_file2(train_data_path,batch_size = 64,col = col),steps_per_epoch=105000//64,epochs=3,validation_data = generate_arrays_from_file(validate_data_path,batch_size = 64,col = col),validation_steps = 15000//64)
    model1.save('./model_2/'+col+'.model')
#1    pdb.set_trace()
    predictions = model1.predict(test_x)
    predictions1 = model1.predict(val_x)
#    pdb.set_trace()
    predictions1 = np.argmax(predictions1, axis=1)
    predictions = np.argmax(predictions, axis=1)
    val_y = data[col]+2
    val_y = np.array(val_y)
    F1_score = f1_score(predictions1, val_y, average='macro')
    F1_scores += F1_score
#1    logger.info(col+' f1_score: '+str(F1_score)+' ACC_score: '+str(accuracy_score(predictions1, val_y)))
#    model1.save_weights("./model/model"+y_cols[index]+'.h5')
#1    model1.save_weights("./model/model"+col+'.h5')
    K.clear_session()
#    sub = pd.DataFrame({y_cols[index]:predictions})
#    sub.to_csv('./TCNN_pre/'+y_cols[index]+'.csv',index = False)
#1    sub = pd.DataFrame({col:predictions})
#1    sub.to_csv('./TCNN_pre/'+col+'.csv',index = False)

logger.info('all F1_score:'+str(F1_scores/len(y_cols)))



