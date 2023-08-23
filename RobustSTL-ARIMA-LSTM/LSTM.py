import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 读取数据
df = pd.read_csv('D:\desktop\Rstl论文\修改意见\夏威夷月度.csv', parse_dates=['date'], index_col='date')
#print(df.head())
#数据预处理
df=np.log(df)
# 数据可视化
'''plt.figure(figsize=(12, 6))
plt.plot(df)
plt.title('Data')
plt.xlabel('Time')
plt.ylabel('Number')
plt.show()'''

# 数据预处理
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df.values)

# 定义函数，生成X和y
def create_dataset(dataset, look_back=4):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

# 准备训练数据和测试数据
train_size = int(len(scaled_values) * 0.8)
test_size = len(scaled_values) - train_size
train, test = scaled_values[0:train_size, :], scaled_values[train_size:len(scaled_values), :]
print(len(train), len(test))

look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 将输入转换成需要的格式 [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)

# 单步预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

'''# 反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])'''
trainY = [trainY]
testY = [testY]

# 计算RMSE
#trainScore = np.sqrt(np.mean((trainY[0] - trainPredict[:, 0]) ** 2))
#testScore = np.sqrt(np.mean((testY[0] - testPredict[:, 0]) ** 2))
#print('Train Score: %.5f RMSE' % (trainScore))
#print('Test Score: %.5f RMSE' % (testScore))
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from sklearn import metrics
print(testY[0])
print(testPredict[:, 0])
testPredict[:, 0]=np.round(testPredict[:, 0], decimals=2)
print(testPredict[:, 0])
mse = mean_squared_error(testY[0], testPredict[:, 0])

mae = mean_absolute_error(testY[0], testPredict[:, 0])

RMSE = math.sqrt(mse)

MAPE = metrics.mean_absolute_percentage_error(testY[0], testPredict[:, 0])

print('mse:{:.3f}, mae:{:.3f}, rmse:{:.3f},mape:{:.3f}'.format(mse, mae, RMSE,MAPE))
# 单步预测图像
# 单步预测可视化
trainPredictPlot = np.empty_like(scaled_values)
trainPredictPlot[:, :] = np.nan
#trainPredictPlot[look_back:len(trainPredict) + look_back, :] = train
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
testPredictPlot = np.empty_like(scaled_values)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2):len(scaled_values), :] = testPredict

plt.figure(figsize=(12, 6))
#plt.plot(scaler.inverse_transform(scaled_values), label='Actual')
plt.plot(scaled_values, label='Actual')
plt.plot(trainPredictPlot, label='Training Prediction')
plt.plot(testPredictPlot, label='Testing Prediction')
plt.title('Single-Step Prediction')
plt.xlabel('Time')
plt.ylabel('tourist')
plt.legend()
plt.show()


# 多步预测
'''future_inputs = df['number'].values[-look_back:]
future_inputs = future_inputs.reshape(1, -1)

future_predictions = []
for i in range(4):
    prediction = model.predict(future_inputs.reshape(1, look_back, 1))
    future_predictions.append(prediction[0, 0])
    future_inputs = np.roll(future_inputs, -1)
    future_inputs[-1, -1] = prediction[0, 0]'''
future_inputs = scaled_values[-look_back:]
future_inputs = future_inputs.reshape(1, look_back, 1)

future_predictions = []
for i in range(9):
    prediction = model.predict(future_inputs)
    future_predictions.append(prediction[0, 0])
    future_inputs = np.concatenate((future_inputs[:, 1:, :], prediction.reshape(1, 1, 1)), axis=1)



# 反归一化
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# 绘制图像
future_index = pd.date_range(df.index[-1], periods=len(future_predictions), freq='15T')

future_df = pd.DataFrame({'tourist': future_predictions.flatten()}, index=future_index)
future_df.index.name = 'time'

plt.figure(figsize=(12, 6))
plt.plot(df['tourist'])
plt.plot(future_df['tourist'])
plt.title('Multi-Step Prediction')
plt.xlabel('Time')
plt.ylabel('Number')
plt.legend(['Original Data', 'Future Predictions'])
plt.show()

# 多步预测误差计算
actual_values = df.values[-9:]
predicted_values = future_predictions[-9:]
#mse = np.mean((actual_values - predicted_values)**2)
#print(f'MSE for 10-step prediction: {mse}')


mse = mean_squared_error(actual_values, predicted_values)

RMSE = math.sqrt(mse)

mae = mean_absolute_error(actual_values, predicted_values)
'''def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))
MAPE=MAPE(predicted_values,actual_values)
print(MAPE)'''
MAPE = metrics.mean_absolute_percentage_error(actual_values, predicted_values)

print('mse:{:.3f}, mae:{:.3f}, rmse:{:.3f},mape:{:.3f}'.format(mse, mae, RMSE,MAPE))
#print(actual_values)
#print(predicted_values)

'''def mean_absolute_scaled_error(actual_values, predicted_values):
    """
    计算平均绝对标度误差（MASE）

    参数：
    y_true: 实际观测值的数组（一维）
    y_pred: 预测值的数组（一维）

    返回：
    mase: 平均绝对标度误差的值
    """
    n = len(actual_values)
    if n != len(predicted_values):
        raise ValueError("y_true and y_pred must have the same length.")

    mae = sum(abs(actual_values[i] - predicted_values[i]) for i in range(n)) / n

    scaling_factor = 1 / ((n - 1) * mae)
    mase = scaling_factor * sum(abs(actual_values[i] - predicted_values[i]) for i in range(1, n)) / n

    return mase'''

# 示例数据


# 计算MASE
'''mase_value = mean_absolute_scaled_error(actual_values, predicted_values)
print("MASE:", mase_value)
'''