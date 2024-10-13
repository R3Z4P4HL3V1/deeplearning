import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# import dataset
data = pd.read_csv('water_potability.csv')

# menampilkan informasi pada dataset
data.info()
print()

# menghitung dan menampilkan berapa banyak nilai hilang
missing_data1 = data.isnull().sum()
print(missing_data1)
print()

# menghilangkan baris data yang hilang
data.dropna(axis=0,inplace=True)
missing_data2 = data.isnull().sum()
print(missing_data2)
print()

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Splitting dataset to test set dan training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x_train)
print(y_train)
print()
print(x_test)
print(y_test)
print()

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print("------------------------------")
print(x_test)
print()

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)