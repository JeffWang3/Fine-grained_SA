# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:39:49 2018

@author: Jeff
"""

import pandas as pd

train_data_path = "./train_data/sentiment_analysis_trainingset.csv"  
validate_data_path = "./train_data/sentiment_analysis_validationset.csv"
test_data_path = "./train_data/sentiment_analysis_testa.csv"

def load_data_from_csv(file_name, header=0, encoding="utf-8"):

    data_df = pd.read_csv(file_name, header=header, encoding=encoding)

    return data_df

train = load_data_from_csv(train_data_path)
val = load_data_from_csv(validate_data_path)
test = load_data_from_csv(test_data_path)

train.iloc[1:1001,:].to_csv("./train_data/small_train.csv",encoding="utf-8")
val.iloc[1:501,:].to_csv("./train_data/small_val.csv",encoding="utf-8")
test.iloc[1:501,:].to_csv("./train_data/small_test.csv",encoding="utf-8")