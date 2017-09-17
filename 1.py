from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt

x,y = make_regression(n_features=1,noise=10)


ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))


model = LinearRegression()
train_x,test_x,train_y,test_y = train_test_split(x,y)
plt.scatter(train_x,train_y,color="red",marker='o')
plt.scatter(test_x,test_y,color="blue",marker='x')
model.fit(train_x,train_y)
prediction = model.predict(test_x)
plt.scatter(test_x,prediction,color='green')


plt.show()

