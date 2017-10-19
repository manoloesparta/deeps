def preprocessing(dataframe):
	
	import pandas as pd

	X = dataframe.drop(['CustomerId','RowNumber','Surname','Exited'],axis=1)
	y = dataframe['Exited']

	from sklearn.preprocessing import LabelEncoder

	le = LabelEncoder()
	gender = le.fit_transform(dataframe['Gender'])

	geo = pd.get_dummies(dataframe['Geography'],drop_first=True)

	X = X.drop(['Gender','Geography'],axis=1)
	X['Gender'] = gender

	X = pd.concat([X,geo],axis=1)

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.fit_transform(X_test)

	print("Preprocessing complete!")

	return [X_train,y_train,X_test,y_test,scaler]


if __name__ == "__main__":

	import pandas as pd

	dataframe = pd.read_csv("Churn_Modelling.csv")

	pre = preprocessing(dataframe)

	print(pre)