import numpy as np
import pandas as pd
from google.colab import files
uploaded = files.upload()
import io
iris = pd.read_csv(io.BytesIO(uploaded['iris.csv']))
print(iris)
print(iris.shape)
print(iris.descibe)
x=iris.drop(columns=['class'])
print(x)
print(x.shape)
y=iris['class']
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.4)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
from sklearn import neighbors
classifier1=neighbors.KNeighborsClassifier()
k=classifier1.fit(x_train,y_train)
print(k)
predictions=classifier1.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
from sklearn import tree
classifier2=tree.DecisionTreeClassifier()
classifier2.fit(x_train,y_train)
predictions=classifier2.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


