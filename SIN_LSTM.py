import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 生成数据
x_len = 1075
x = np.linspace(0, np.pi * 10.75, x_len, endpoint=False)
y = np.cos(x).reshape(-1, 1)
print(type(y))
print("显示生成的余弦数据：")
#plt.scatter(x, y)
#plt.show()

window = 75  # 时序滑窗大小，就是time_steps
batch_size = 256
tg = TimeseriesGenerator(y, y, length=window, batch_size=batch_size)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(
    units=50, input_shape=(window, 1), return_sequences=True # True返回输出序列的全部
))
model.add(LSTM(
    units=100, return_sequences=False # False返回输出序列的最后一个
))
model.add(Dense(1)) # 输出层
model.compile(optimizer='adam', loss='mse')  # 均方误差（Mean Square Error）
model.fit_generator(generator=tg, epochs=30)

# 预测
pred_len = 200  # 预测序列长度
for i in range(4):
    x_pred = x[i * batch_size + window: i * batch_size + window + pred_len]
    y_pred = []  # 存放拟合序列
    X_pred = tg[i][0][0]
    for i in range(pred_len):
        Y_pred = model.predict(X_pred.reshape(-1, window, 1))  # 预测
        y_pred.append(Y_pred[0])
        X_pred = np.concatenate((X_pred, Y_pred))[1:]  # 窗口滑动
    plt.scatter(x_pred[0], y_pred[0], c='r', s=9)  # 预测起始点
    plt.plot(x_pred, y_pred, 'r')  # 预测序列
#plt.plot(x, y, 'y', linewidth=5, alpha=0.3)  # 原序列
plt.show()
