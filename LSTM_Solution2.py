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
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv('pollution.csv',usecols=['dew', 'temp', 'press','wnd_dir', 'wnd_spd', 'snow', 'rain'])
target=pd.read_csv('pollution.csv',usecols=['pollution'])
encoder = LabelEncoder()
df=df.values
df[:,3] = encoder.fit_transform(df[:,3])
#print(df.head())
#df.date=pd.to_datetime(df.date)
#df=df.set_index('date')
#print(df.head())
'''
#train,test=df[:-12],df[-12:]
#train_target,test_target=target[:-12],target[-12:]
scaler=MinMaxScaler()
scaler.fit(train)
train =scaler.transform(train)
test =scaler.transform(test)
'''
scaler=MinMaxScaler()
scaler.fit(df)
df =scaler.transform(df)
#df=df.values
target=target.values
#test =scaler.transform(test)
print(type(df))
print(type(target))
n_input=6
n_features=7
generator=TimeseriesGenerator(df,target,length=n_input,batch_size=1)
for i in range(2):
	x, y = generator[i]
	print('%s => %s' % (x, y))

model=Sequential()
model.add(LSTM(200,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit_generator(generator, epochs=180)
