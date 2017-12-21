from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import numpy as np

# You can play with these parameters
endNum = 10000
windowSize = 1000
numEpochs = 1
numBatchSize = 128

sineWave = np.sin(np.linspace(0, 10*np.pi, endNum))

# Split into overlapping windows of 50 timesteps
waveWindow = np.zeros((endNum-windowSize, windowSize))
for nStep in range(0, endNum-windowSize):
    waveWindow[nStep, :] = sineWave[nStep:nStep+windowSize]
    
# Dimensions of x are [num_windows, windows, predictors]
x = np.expand_dims(waveWindow, 2)
y = sineWave[0:waveWindow.shape[0]]

# Let's run a quick neural network
model = Sequential()

# Layer 1
model.add(LSTM(1, return_sequences=True, input_shape=(windowSize, 1)))
model.add(Dropout(0.2))

# Layer 2
model.add(LSTM(50))
model.add(Dropout(0.2))

# Layer 3
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Fit the model, this is the slowest part.
model.fit(x, y, batch_size=numBatchSize, epochs=numEpochs, verbose=1, shuffle=True)

# Make some predictions
predictions = model.predict(x)
