from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset1 = read_csv('./train_data/1.csv', header=0, index_col=0)
dataset2 = read_csv('./train_data/2.csv', header=0, index_col=0)
dataset3 = read_csv('./train_data/3.csv', header=0, index_col=0)
dataset4 = read_csv('./train_data/4.csv', header=0, index_col=0)
dataset9 = read_csv('./train_data/9.csv', header=0, index_col=0)

values1 = dataset1.values
values2 = dataset2.values
values3 = dataset3.values
values4 = dataset4.values
values9 = dataset9.values

values1 = values1.astype('float32')
values2 = values2.astype('float32')
values3 = values3.astype('float32')
values4 = values4.astype('float32')
values9 = values9.astype('float32')
np.set_printoptions(threshold=np.inf)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled1 = scaler.fit_transform(values1[:,:-1])
scaled1=np.c_[scaled1,values1[:,-1:]]

scaled2 = scaler.fit_transform(values2[:,:-1])
scaled2=np.c_[scaled2,values2[:,-1:]]
# print(values2[0:1])
# print(scaled2[0:1])
scaled3 = scaler.fit_transform(values3[:,:-1])
scaled3=np.c_[scaled3,values3[:,-1:]]
scaled4 = scaler.fit_transform(values4[:,:-1])
scaled4=np.c_[scaled4,values4[:,-1:]]
scaled9 = scaler.fit_transform(values9[:,:-1])
scaled9=np.c_[scaled9,values9[:,-1:]]
# specify the number of lag hours
n_hours = 12
n_features = 16
# frame as supervised learning
reframed1 = series_to_supervised(scaled1, n_hours, 1)
reframed2 = series_to_supervised(scaled2, n_hours, 1)
reframed3 = series_to_supervised(scaled3, n_hours, 1)
reframed4 = series_to_supervised(scaled4, n_hours, 1)
reframed9 = series_to_supervised(scaled9, n_hours, 1)

#reframed = TimeseriesGenerator(scaled, n_hours, 1)
#print(reframed1[0:1])

# split into train and test sets
train1 = reframed1.values
train2 = reframed2.values
train3 = reframed3.values
train4 = reframed4.values
train9 = reframed9.values
# split into input and outputs
n_obs = n_hours * n_features
#print(train1[0:1])
train1_X, train1_y = train1[:, :n_obs], train1[:, -1]

#print(train1_X[0:1])
#print(train1_y[0:1])
train2_X, train2_y = train2[:, :n_obs], train2[:, -1]
train3_X, train3_y = train3[:, :n_obs], train3[:, -1]
train4_X, train4_y = train4[:, :n_obs], train4[:, -1]
train9_X, train9_y = train9[:, :n_obs], train9[:, -1]
#print(train1_X.shape, len(train1_X), train1_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train1_X = train1_X.reshape((train1_X.shape[0], n_hours, n_features))
train2_X = train2_X.reshape((train2_X.shape[0], n_hours, n_features))
train3_X = train3_X.reshape((train3_X.shape[0], n_hours, n_features))
train4_X = train4_X.reshape((train4_X.shape[0], n_hours, n_features))
train9_X = train9_X.reshape((train9_X.shape[0], n_hours, n_features))

test_X,test_y=train9_X,train9_y
#print(train_X.shape, len(train_X), train_y.shape)
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train1_X.shape[1], train1_X.shape[2]),return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train1_X, train1_y, epochs=50, batch_size=40, validation_data=(train4_X,train4_y), verbose=2, shuffle=False)
model.save('my_model.h5')



model= tf.keras.models.load_model('my_model.h5')
history = model.fit(train2_X, train2_y, epochs=50, batch_size=40, validation_data=(train4_X,train4_y), verbose=2, shuffle=False)
model.save('my_model.h5')
model= tf.keras.models.load_model('my_model.h5')
history = model.fit(train3_X, train3_y, epochs=50, batch_size=40, validation_data=(train4_X,train4_y), verbose=2, shuffle=False)
model.save('my_model.h5')
model= tf.keras.models.load_model('my_model.h5')
history = model.fit(train4_X, train4_y, epochs=50, batch_size=40, validation_data=(train4_X,train4_y), verbose=2, shuffle=False)
model.save('my_model.h5')
# continue fitting

# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# make a prediction
# plot each column
yhat = model.predict(test_X)
pyplot.figure()
pyplot.plot(test_y)
pyplot.plot(yhat)
pyplot.show()

# test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
