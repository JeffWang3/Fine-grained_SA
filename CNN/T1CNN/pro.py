import pandas as pd

k = pd.read_csv("/data1/hjw/AI_Challenger_2018/Baselines/sentiment_analysis2018_baseline/T1CNN/final_result.csv",header=0,encoding="utf-8")

k = k.iloc[:,1:]
k.to_csv("./final_result3.csv")

