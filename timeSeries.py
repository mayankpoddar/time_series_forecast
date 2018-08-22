from keras.models import Model
from keras.layers import Input, LSTM, Dense

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

dataset = np.array(pd.read_csv(r'./international-airline-passengers.csv').values[:, 1], dtype=np.float32)
stationaryDataset = np.diff(np.log(dataset))

# plt.plot(stationaryDataset)
lookback = 30

X = np.zeros([len(stationaryDataset) - lookback - 1, lookback], dtype=np.float32)
Y = np.zeros([len(stationaryDataset) - lookback - 1, 1], dtype=np.float32)

for i in range(lookback):
    X[:, i] = stationaryDataset[i:i-lookback-1]
X = X.reshape([-1, lookback, 1])
Y[:, 0] = stationaryDataset[lookback:-1]

inputLayer = Input([lookback, 1])
lstmLayer = LSTM(lookback*2, dropout=0.1, recurrent_dropout=0.1, activation='tanh')(inputLayer)
outputLayer = Dense(1, activation='tanh')(lstmLayer)

model = Model(inputLayer, outputLayer)
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())
model.fit(X, Y, batch_size=1, epochs=500)

prediction = np.array(X.shape[0]*[np.nan], dtype=np.float32)
results = []
toPred = X[-1].reshape([1, lookback, 1])
for i in range(100):
    pred = model.predict(toPred)
    toPred[0][:-1] = toPred[0][1:]
    toPred[0][-1] = pred
    results.append(pred)

results = np.array(results, dtype=np.float32).reshape([-1, ])
trend = model.predict(X).reshape([-1, ])
plt2 = np.exp(np.concatenate(([np.log(dataset[0])], trend)).cumsum())
results = np.exp(np.concatenate(([np.log(plt2[-1])], results)).cumsum())
prediction = np.append(prediction, results)

plt.plot(dataset)
plt.plot(plt2, color='red')
plt.plot(prediction, color='green')
plt.savefig('dataset.png')

model.save('./model.hdf5')
