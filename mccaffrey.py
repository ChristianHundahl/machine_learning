from re import X
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

model = Sequential() #empty model, no data before .add
model.add(Dense(8, input_dim=6, activation='sigmoid')) #Adding hidden layer: (Dense(hidden_layer_neuroner, input_dim=input_dimensions, activation=function_used)
model.add(Dense(3, activation='sigmoid')) #Adding output: outputs=3
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)

#each test data (person) exprressed as numbers
person1 = [0.27, 0.24, 1, 1, 0, 0]
person2 = [0.48, 0.98, -1, 0, 1, 0]
person3 = [0.33, 0.44, -1, 0, 0, 1]
person4 = [0.33, 0.44, -1, 0, 0, 1]
person5 = [0.3, 0.29, 1, 0, 1, 0]
person6 = [0.66, 0.65, -1, 1, 0, 0]

x = np.array([person1, person2, person3, person4, person5, person6]) 

y = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0,0,1], [1,0,1], [1,0,0]])

history = model.fit(x,y,epochs=2000, batch_size=2, verbose=1) #verbose = 1 shows output, 0 no output shown

prediction = model.predict([[0.38, 0.51, 1, 1, 0, 0]])
print(prediction)