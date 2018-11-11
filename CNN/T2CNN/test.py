import keras as ks
import pdb
from sklearn.metrics import f1_score,accuracy_score
import numpy as np

model = ks.models.load_model('./model_2/location_traffic_convenience.model')
#pdb.set_trace()

test = np.load('val.npy')
test = np.reshape(test,(-1,300,300))
val_p = model.predict(test)
#val_p = np.load('./val_p.npy')
val_y = np.load('./val_y.npy')

def precission(val_p,val_y,flag):
    tp = 0 #true_positive
    fn = 0 #false_negative
    fp = 0 #false_positive
    tn = 0   
    
    for i,j in zip(val_p,val_y):
        if j == flag: #pre positive
            if i == flag: 
                tp += 1
            else:
                fn += 1
        else:
            if i == flag:
                fp += 1
            else:
                tn += 1
    return (tp,fn,fp,tn)

pdb.set_trace()
