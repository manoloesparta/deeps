def train_model(X_train,y_train,batch_size,nb_epoch,file):
	
	import keras
	from keras.models import Sequential
	from keras.layers import Dense
	from save_model import save_model as sm

	model = Sequential()

	model.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
	model.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

	model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

	try:

		model.fit(X_train,y_train,batch_size,nb_epoch)

	except:

		sm(model,file)
		return model

	sm(model,file)
	return model

if __name__ == "__main__":

	import pandas as pd
	from preprocessing import preprocessing as pp

	dataframe = pd.read_csv("Churn_Modelling.csv")

	pre = pp(dataframe)

	X_train = pre[0]
	y_train = pre[1]

	train_model(X_train,y_train,10,1,"archivo")

	print("Training complete!")