def one_sample():

	import pandas as pd
	import numpy as np

	from preprocessing import preprocessing

	data = pd.read_csv("Churn_Modelling.csv")
	pre = preprocessing(data)

	scaler = pre[-1]

	geo = input("Geography: ")
	score = int(input("Credit Score:"))
	gender = input("Gender (m/f): ")
	age = int(input("Age: "))
	tenure = int(input("Tenure (year): "))
	balance = int(input("Balance: "))
	num_pro = int(input("Number of Procuts: "))
	credit = input("Does it have credit card (y/n): ")
	active = input("Is this custmoer active memeber (y/n): ")
	salary = int(input("Estimated Salary: "))

	geo = geo.lower()
	gender = gender.lower()
	credit = credit.lower()
	active = active.lower()

	if geo == "france":
		geo = 0
		geo1 = 0
	elif geo == "spain":
		geo = 1
		geo1 = 0
	elif geo == "germany":
		geo = 0
		geo1 = 1

	if gender == "f":
		gender = 0
	elif gender == "m":
		gender = 1

	if credit == "y":
		credit = 1
	elif credit == "n":
		credit = 0

	if active == "y":
		active = 1
	elif active == "n":
		active = 0

	arr = np.array([[float(geo),geo1,score,gender,age,tenure,balance,num_pro,credit,active,salary]])
	arr = scaler.transform(arr)

	return arr

if __name__ == "__main__":

	arr = one_sample()
	print(arr)