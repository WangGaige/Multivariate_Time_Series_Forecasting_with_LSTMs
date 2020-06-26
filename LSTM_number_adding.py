from numpy import array
from numpy import hstack
from numpy import insert
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import  rmse
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
# 给定数据
in_seq1 =array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([25, 45, 65, 85, 105, 125, 145, 165, 185, 205])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
print(type(in_seq1))
in_seq1= in_seq1.astype('float32')
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq2= in_seq1.astype('float32')
out_seq = out_seq.reshape((len(out_seq), 1))
out_seq= out_seq.astype('float32')
# 按行的方向进行堆叠
dataset = hstack((in_seq1, in_seq2))
#print(dataset)
# 插入一个值作为初始值
#out_seq = insert(out_seq, 0, 0)
# 定义生成器
n_input = 2

generator = TimeseriesGenerator(dataset, out_seq, length=n_input, batch_size=2)

model=Sequential()
model.add(LSTM(200,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(generator, epochs=180)