import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(units=1, activation='linear'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)

heights = np.array([[75], [92], [108], [121], [130], [142], [155]])
ages = np.array([[1], [3], [5], [7], [9], [11], [13]])

history = model.fit(ages, heights, epochs=1000, batch_size=2, verbose=1)

prediction = model.predict([70])
print(prediction)