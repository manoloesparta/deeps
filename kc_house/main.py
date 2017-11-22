import numpy as np
import pandas as pd

kc = pd.read_csv('kc_house_data.csv')

X = kc.drop(['id','price','date'],axis=1)
y = kc['price']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

model = Sequential()

model.add(Dense(units=9,kernel_initializer='uniform',activation='relu',input_dim=18))
model.add(Dropout(p=0.1))
model.add(Dense(units=9,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(p=0.1))

model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=10,nb_epoch=100,verbose=1)