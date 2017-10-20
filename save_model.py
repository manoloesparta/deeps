def save_model(model,file):

	model_json = model.to_json()
	with open(file + ".json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights(file + ".h5")
	print("Saved model to disk")

if __name__ == "__main__":

	import pandas as pd
	from preprocessing import preprocessing as pp
	from train import train_model

	dataframe = pd.read_csv("Churn_Modelling.csv")

	pre = pp(dataframe)

	X_train = pre[0]
	y_train = pre[1]

	model = train_model(X_train,y_train,10,1,'model')

	save_model(model,'model')