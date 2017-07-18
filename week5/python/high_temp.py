import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('daily-maximum-temperatures-in-me.csv')

raw_data = df.max_temp.values

scaler = MinMaxScaler(feature_range=(0,1))
raw_data = scaler.fit_transform(raw_data.reshape(-1,1))

timestep = 5
X = []
Y = []
for i in range(len(raw_data)-(timestep)):
    X.append(raw_data[i:i+timestep])
    Y.append(raw_data[i+timestep])

X = np.asanyarray(X)
X = X.reshape((X.shape[0],X.shape[1],1))

Y = np.asanyarray(Y)

k = int(0.7*len(Y))
Xtrain = X[:k, :, :]
Xtest = X[k:, :, :]

Ytrain = Y[:k]
Ytest = Y[k:]

model = Sequential()
model.add(LSTM(64,batch_input_shape=(None, timestep, 1),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(Xtrain,
          Ytrain,
          batch_size=20,
          validation_data=(Xtest, Ytest),
          verbose=1,
          epochs=30,
          shuffle=False)


rand_pos = random.randint(0, len(Ytest)-100)
forcasted_output = []
Xin = Xtest[rand_pos:rand_pos+1, :, :]

selected_input = Xtest[rand_pos:rand_pos+100, :, :]
real_output = scaler.inverse_transform(Ytest[rand_pos:rand_pos+100])
predicted_output = model.predict(selected_input, batch_size=1)
predicted_output = scaler.inverse_transform(predicted_output)

print('Plotting Results')
plt.figure(figsize=(12,5))
xpos = range(len(predicted_output))

plt.plot(xpos, real_output,'r', xpos, predicted_output, 'b')
plt.legend(('Real', 'Predicted'))
plt.show()