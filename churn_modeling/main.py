import pandas as pd
import numpy as np

from train import train_model
from save_model import save_model
from load_model import load_model
from preprocessing import preprocessing
from one_sample import one_sample

from pathlib import Path

from sklearn.metrics import classification_report

dataset = pd.read_csv("Churn_Modelling.csv")
X_train,y_train,X_test,y_test,scaler = preprocessing(dataset)

load = int(input("Load Model (0/1): "))

if load == 1:

	file = input("File (no extensions): ")

	if not Path(file + '.json').is_file():
		print("File not found")
		load = 0
	else:
		model = load_model(file)

if load == 0:

	print("Creating neural network...")

	file = input("Name: ")
	epochs = int(input("Epcohs: "))
	batch = int(input("Batch Size: "))

	model = train_model(X_train,y_train,batch,epochs,file)


more_train = int(input("More training (0/1): "))

if more_train == 1:

	more_epochs = int(input("How much epochs: "))
	batch = int(input("Batch Size: "))
	model = train_model(X_train,y_train,batch,more_epochs,file)


predict = int(input("Predict y_test (0/1): "))

y_pred = model.predict(X_test)
y_pred_bool = (y_pred > 0.5)

if predict == 1:

	print("Probability of each one to quit, sample = 10")
	for i in y_pred[0:10]:
		print(i)

	print("They will quit in the next 6 months?, sample = 10")
	for i in y_pred_bool[0:10]:
		print(i)

medir = int(input("Check precision (0/1): "))

if medir == 1:

	accuracy = classification_report(y_test,y_pred_bool)
	print(accuracy)

tryit = int(input("Predict over one sample (0/1): "))

if tryit == 1:

	pred = model.predict(one_sample())
	print("Probability to get out")
	print(pred)

	pred = (pred > 0.5)
	print("He will quit in the next 6 months?")
	print(pred)

