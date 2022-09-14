import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)

x = np.array([[0,0],[1,0],[0,1],[1,1]]) #(Dense(hidden_layer_neuroner, input_dim=input_dimensions, activation=function_used)

xor_test = np.array([[0], [1], [1], [0]]) #XOR
and_test = np.array([[0], [0], [0], [1]]) #AND
or_test  = np.array([[0], [1], [1], [1]]) #OR

history = model.fit(x,and_test,epochs=1000, batch_size=2, verbose=0) #verbose = 1 shows output, 0 no output shown

#and
print('AND')
prediction_1_1 = model.predict([[1,1]])
prediction_0_1 = model.predict([[0,1]])
prediction_1_0 = model.predict([[1,0]])
prediction_0_0 = model.predict([[0,0]])
print(prediction_1_1)
print(prediction_0_1)
print(prediction_1_0)
print(prediction_0_0)

#xor
print('XOR')
history = model.fit(x, xor_test, epochs=2000, batch_size=2, verbose=0)
prediction_1_1 = model.predict([[1,1]])
prediction_0_1 = model.predict([[0,1]])
prediction_1_0 = model.predict([[1,0]])
prediction_0_0 = model.predict([[0,0]])
print(prediction_1_1)
print(prediction_0_1)
print(prediction_1_0)
print(prediction_0_0)

#or
print('OR')
history = model.fit(x, or_test, epochs=2000, batch_size=2, verbose=0)
prediction_1_1 = model.predict([[1,1]])
prediction_0_1 = model.predict([[0,1]])
prediction_1_0 = model.predict([[1,0]])
prediction_0_0 = model.predict([[0,0]])
print(prediction_1_1)
print(prediction_0_1)
print(prediction_1_0)
print(prediction_0_0)
