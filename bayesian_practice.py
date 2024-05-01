import pandas as pd  
import numpy as np
import sklearn as sk
from sklearn import naive_bayes
import matplotlib as plt 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
df = pd.read_csv(r"C:\Users\miroe\Downloads\archive(1)\Naive-Bayes-Classification-Data.csv")
df.head()


x = df["glucose"]
y = df["bloodpressure"]
z = df["diabetes"]
target = df.diabetes
input = df.drop("diabetes", axis = "columns")

x_train, x_test, y_train, y_test = train_test_split(input, target, test_size = 0.3)
model = GaussianNB()
model.fit(x_train, y_train)
model.score(x_test, y_test)
y_test[:10]
model.predict(x_test [:10])

