import numpy as np
import pandas as pd

import tensorflow as tf

# encoding geography column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#  encoding gender column
from sklearn.preprocessing import LabelEncoder

# splitting dataset into the training set and test set
from sklearn.model_selection import train_test_split

# feature scalling
from sklearn.preprocessing import StandardScaler

# confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score

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

# splitting dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# implementation feature Scalling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Building ANN
# Inisialisasi Artificial Neural Network
ann = tf.keras.models.Sequential()

# adding input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) # 6 merupakan hyper parameter, tidak ada aturan untuk mengubahnya

# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# adding output layer
# mengapa kode pada output layer sama?, karena kita harus tetap menambahkan layer baru untuk output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# training the Model
# compile ANN with optimizer, loss function, and metric
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

